import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation
from losses import DiceLoss, FocalLoss, LovaszLoss
from config import DEVICE
 
# ================= Utils =================
def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    if drop_prob == 0. or not training: return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep: random_tensor.div_(keep_prob)
    return x * random_tensor

class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep
    def forward(self, x): return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

# ================= ConvNeXt Modules =================
class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, drop_path=0.1):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.GroupNorm(1, dim)
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(4 * dim, dim, kernel_size=1)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    def forward(self, x):
        input = x; x = self.dwconv(x); x = self.norm(x); x = self.pwconv1(x)
        x = self.act(x); x = self.pwconv2(x); x = input + self.drop_path(x)
        return x

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=bias)
        self.bn = nn.GroupNorm(1, out_planes) if bn else nn.Identity()
        self.relu = nn.GELU() if relu else nn.Identity()
    def forward(self, x): return self.relu(self.bn(self.conv(x)))

class DownsampleLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.norm = nn.GroupNorm(1, in_dim)
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=2, stride=2, bias=False)
    def forward(self, x): return self.conv(self.norm(x))

class AirQualityEncoder_ConvNeXt(nn.Module):
    def __init__(self, in_channels=48, channels_list=[64, 128, 320, 512], drop_path_rate=0.1, num_blocks=2):
        super(AirQualityEncoder_ConvNeXt, self).__init__()
        C0, C1, C2, C3 = channels_list[0], channels_list[1], channels_list[2], channels_list[3]
        self.stem_conv = BasicConv(in_channels, C1, kernel_size=3, stride=1, padding=1)
        self.stem_blocks = nn.Sequential(*[ConvNeXtBlock(C1, drop_path=drop_path_rate) for _ in range(num_blocks)])
        self.make_f1_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.make_f1_conv = BasicConv(C1, C0, kernel_size=3, stride=1, padding=1)
        self.make_f1_blocks = nn.Sequential(*[ConvNeXtBlock(C0, drop_path=drop_path_rate) for _ in range(num_blocks)])
        self.make_f3_downsample = DownsampleLayer(C1, C2)
        self.make_f3_blocks = nn.Sequential(*[ConvNeXtBlock(C2, drop_path=drop_path_rate) for _ in range(num_blocks)])
        self.make_f4_downsample = DownsampleLayer(C2, C3)
        self.make_f4_blocks = nn.Sequential(*[ConvNeXtBlock(C3, drop_path=drop_path_rate) for _ in range(num_blocks)])
        print(f"--- [ConvNeXt] AirEncoder Build Complete ---")

    def forward(self, x):
        f2_stem = self.stem_conv(x)
        f2 = self.stem_blocks(f2_stem)
        f1_up = self.make_f1_upsample(f2)
        f1_conv = self.make_f1_conv(f1_up)
        f1 = self.make_f1_blocks(f1_conv) 
        f3_down = self.make_f3_downsample(f2)
        f3 = self.make_f3_blocks(f3_down) 
        f4_down = self.make_f4_downsample(f3)
        f4 = self.make_f4_blocks(f4_down)
        return [f1, f2, f3, f4]

# ================= SegFormer Encoder Helper =================
def get_segformer_encoder(num_channels=4, model_id="nvidia/segformer-b1-finetuned-ade-512-512"):
    model = SegformerForSemanticSegmentation.from_pretrained(model_id, num_labels=1, ignore_mismatched_sizes=True)
    original_conv = model.segformer.encoder.patch_embeddings[0].proj
    old_weights = original_conv.weight.clone()
    new_conv = nn.Conv2d(num_channels, original_conv.out_channels, kernel_size=original_conv.kernel_size, stride=original_conv.stride, padding=original_conv.padding)
    with torch.no_grad():
        new_conv.weight[:, :3, :, :] = old_weights
        new_conv.weight[:, 3:, :, :] = old_weights.mean(1, keepdim=True)
        if original_conv.bias is not None: new_conv.bias = nn.Parameter(original_conv.bias.clone())
    model.segformer.encoder.patch_embeddings[0].proj = new_conv
    print(f"--- Segformer (B1) Encoder Build Complete (In: {num_channels}) ---")
    return model.segformer, model.decode_head

# ================= Fusion Modules =================
class FiLMLayer(nn.Module):
    def __init__(self, air_dim, vis_dim):
        super().__init__()
        mid_dim = (air_dim + vis_dim) // 2
        self.generator = nn.Sequential(
            nn.Conv2d(air_dim, mid_dim, kernel_size=1),
            ConvNeXtBlock(mid_dim), 
            nn.Conv2d(mid_dim, vis_dim * 2, kernel_size=1)
            )
    def forward(self, f_vis, f_air):
        gamma_beta = self.generator(f_air)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)
        fused_film = f_vis * (gamma + 1) + beta
        return fused_film

class DynamicFusionGate(nn.Module):
    def __init__(self, vis_dim, air_dim):
        super().__init__()
        mid_dim = vis_dim // 2
        self.gate_generator = nn.Sequential(
            nn.Conv2d(vis_dim + air_dim, mid_dim, kernel_size=3, padding=1),
            nn.GroupNorm(1, mid_dim), nn.GELU(),
            nn.Conv2d(mid_dim, 1, kernel_size=1), nn.Sigmoid()
        )
    def forward(self, f_vis, f_air):
        combined_features = torch.cat([f_vis, f_air], dim=1)
        gate = self.gate_generator(combined_features)
        return gate

class CrossAttentionBlock(nn.Module):
    def __init__(self, vis_dim, air_dim, num_heads=8, ffn_expansion=4, drop_path=0.1):
        super().__init__()
        self.vis_dim = vis_dim
        self.attn = nn.MultiheadAttention(embed_dim=vis_dim, kdim=air_dim, vdim=air_dim, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.GroupNorm(1, vis_dim)
        self.ffn = nn.Sequential(
            nn.Conv2d(vis_dim, vis_dim * ffn_expansion, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(vis_dim * ffn_expansion, vis_dim, kernel_size=1)
        )
        self.norm2 = nn.GroupNorm(1, vis_dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, f_vis, f_air):
        B, C_vis, H, W = f_vis.shape
        vis_flat = f_vis.flatten(2).permute(0, 2, 1)
        air_flat = f_air.flatten(2).permute(0, 2, 1)
        attn_output, _ = self.attn(vis_flat, air_flat, air_flat)
        vis_res1 = (vis_flat + self.drop_path(attn_output)).permute(0, 2, 1).reshape(B, C_vis, H, W)
        vis_norm1 = self.norm1(vis_res1)
        ffn_output = self.ffn(vis_norm1)
        vis_res2 = vis_norm1 + self.drop_path(ffn_output)
        output = self.norm2(vis_res2)
        return output

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=False), nn.GroupNorm(1, F_int))
        self.W_x = nn.Sequential(nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=False), nn.GroupNorm(1, F_int))
        self.act = nn.GELU()
        self.psi = nn.Sequential(nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True), nn.Sigmoid())
    def forward(self, g, x):
        g_out = self.W_g(g)
        x_out = self.W_x(x)
        combined = self.act(g_out + x_out)
        alpha = self.psi(combined) 
        return x * alpha

# ================= Decoder Modules =================
class _ASPPModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, dilation):
        super(_ASPPModule, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation, bias=False),
            nn.GroupNorm(1, out_channels), nn.GELU()
        )
    def forward(self, x): return self.conv(x)

class _ASPPImagePooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(_ASPPImagePooling, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.GroupNorm(1, out_channels), nn.GELU()
        )
    def forward(self, x):
        h, w = x.shape[-2:]; x_pool = self.pool(x); x_conv = self.conv(x_pool)
        x_up = F.interpolate(x_conv, size=(h, w), mode='bilinear', align_corners=False)
        return x_up

class ASPPDecoder(nn.Module):
    def __init__(self, in_channels_high, in_channels_f3, in_channels_low, in_channels_f1, num_classes=1, aspp_out_channels=256, f3_channels=64, low_level_channels=64, f1_channels=64, decoder_channels=[256, 256, 128], num_decoder_blocks=2, drop_path_rate=0.1):
        super(ASPPDecoder, self).__init__()
        f3_refine_dim = decoder_channels[0]
        f2_refine_dim = decoder_channels[1]
        f1_refine_dim = decoder_channels[2]
        rates = [6, 12, 18]
        self.conv_1x1 = _ASPPModule(in_channels_high, aspp_out_channels, 1, padding=0, dilation=1)
        self.atrous_6 = _ASPPModule(in_channels_high, aspp_out_channels, 3, padding=rates[0], dilation=rates[0])
        self.atrous_12 = _ASPPModule(in_channels_high, aspp_out_channels, 3, padding=rates[1], dilation=rates[1])
        self.atrous_18 = _ASPPModule(in_channels_high, aspp_out_channels, 3, padding=rates[2], dilation=rates[2])
        self.image_pool = _ASPPImagePooling(in_channels_high, aspp_out_channels)
        self.concat_conv = nn.Sequential(nn.Conv2d(aspp_out_channels * 5, aspp_out_channels, 1, bias=False), nn.GroupNorm(1, aspp_out_channels), nn.GELU())

        self.f3_conv = nn.Sequential(nn.Conv2d(in_channels_f3, f3_channels, 1, bias=False), nn.GroupNorm(1, f3_channels), nn.GELU())
        self.att_f3 = AttentionGate(F_g=aspp_out_channels, F_l=f3_channels, F_int=f3_channels)
        self.refinement_conv_f3 = nn.Sequential(nn.Conv2d(aspp_out_channels + f3_channels, f3_refine_dim, 1, bias=False), nn.GroupNorm(1, f3_refine_dim), nn.GELU(), *[ConvNeXtBlock(f3_refine_dim, drop_path=drop_path_rate) for _ in range(num_decoder_blocks)])

        self.low_level_conv = nn.Sequential(nn.Conv2d(in_channels_low, low_level_channels, 1, bias=False), nn.GroupNorm(1, low_level_channels), nn.GELU())
        self.att_f2 = AttentionGate(F_g=f3_refine_dim, F_l=low_level_channels, F_int=low_level_channels)
        self.refinement_conv_f2 = nn.Sequential(nn.Conv2d(f3_refine_dim + low_level_channels, f2_refine_dim, 1, bias=False), nn.GroupNorm(1, f2_refine_dim), nn.GELU(), *[ConvNeXtBlock(f2_refine_dim, drop_path=drop_path_rate) for _ in range(num_decoder_blocks)])

        self.f1_conv = nn.Sequential(nn.Conv2d(in_channels_f1, f1_channels, 1, bias=False), nn.GroupNorm(1, f1_channels), nn.GELU())
        self.att_f1 = AttentionGate(F_g=f2_refine_dim, F_l=f1_channels, F_int=f1_channels)
        self.final_refinement_f1 = nn.Sequential(nn.Conv2d(f2_refine_dim + f1_channels, f1_refine_dim, 1, bias=False), nn.GroupNorm(1, f1_refine_dim), nn.GELU(), *[ConvNeXtBlock(f1_refine_dim, drop_path=drop_path_rate) for _ in range(num_decoder_blocks)])

        self.up_conv_1 = nn.Sequential(BasicConv(f1_refine_dim, 64, 3, 1, 1), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        self.up_conv_2 = nn.Sequential(BasicConv(64, 32, 3, 1, 1), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        self.boundary_refine = nn.Sequential(BasicConv(32, 32, kernel_size=3, stride=1, padding=1), ConvNeXtBlock(32, drop_path=drop_path_rate), BasicConv(32, 32, kernel_size=3, stride=1, padding=1))
        
        self.final_classifier_v8 = nn.Conv2d(32, num_classes, 1)
        self.aux_classifier_f3 = nn.Conv2d(f3_refine_dim, num_classes, 1)
        self.aux_classifier_f2 = nn.Conv2d(f2_refine_dim, num_classes, 1)
        print(f"--- [Decoder] Build Complete ---")

    def forward(self, fused_features_list):
        f_low_f1, f_mid_f2, f_mid_f3, f_high_f4 = fused_features_list[0], fused_features_list[1], fused_features_list[2], fused_features_list[3]

        x1, x2, x3, x4, x5 = self.conv_1x1(f_high_f4), self.atrous_6(f_high_f4), self.atrous_12(f_high_f4), self.atrous_18(f_high_f4), self.image_pool(f_high_f4)
        x_aspp = self.concat_conv(torch.cat([x1, x2, x3, x4, x5], dim=1))

        x_aspp_up_f3 = F.interpolate(x_aspp, size=f_mid_f3.shape[-2:], mode='bilinear', align_corners=False)
        x_mid_f3 = self.f3_conv(f_mid_f3)
        x_mid_f3_att = self.att_f3(g=x_aspp_up_f3, x=x_mid_f3)
        x_refine_f3 = self.refinement_conv_f3(torch.cat([x_aspp_up_f3, x_mid_f3_att], dim=1))
        logits_aux_f3 = self.aux_classifier_f3(x_refine_f3)

        x_refine_up_f2 = F.interpolate(x_refine_f3, size=f_mid_f2.shape[-2:], mode='bilinear', align_corners=False)
        x_mid_f2 = self.low_level_conv(f_mid_f2)
        x_mid_f2_att = self.att_f2(g=x_refine_up_f2, x=x_mid_f2)
        x_refine_f2 = self.refinement_conv_f2(torch.cat([x_refine_up_f2, x_mid_f2_att], dim=1))
        logits_aux_f2 = self.aux_classifier_f2(x_refine_f2)

        x_refine_up_f1 = F.interpolate(x_refine_f2, size=f_low_f1.shape[-2:], mode='bilinear', align_corners=False)
        x_low_f1 = self.f1_conv(f_low_f1)
        x_low_f1_att = self.att_f1(g=x_refine_up_f1, x=x_low_f1)
        x_final_refine = self.final_refinement_f1(torch.cat([x_refine_up_f1, x_low_f1_att], dim=1))

        x_up_256 = self.up_conv_1(x_final_refine)
        x_up_512 = self.up_conv_2(x_up_256)      
        x_refined_512 = self.boundary_refine(x_up_512)
        logits_final = self.final_classifier_v8(x_up_512) #same miou with refinement module or not (we can ensenble those two model)

        return {"final": logits_final, "aux_f2": logits_aux_f2, "aux_f3": logits_aux_f3}

# ================= Main Model =================
class DualStream_DFormer_Model(nn.Module):
    def __init__(self, num_classes=2, vis_channels=4, air_channels_input=48, air_channels_list=[64, 128, 320, 512], segformer_id="nvidia/segformer-b1-finetuned-ade-512-512", focal_alpha=0.25, focal_gamma=2.0, lov_weight=0.5, focal_weight=0.5, decoder_channels=[256, 256, 128], num_decoder_blocks=2, drop_path_rate=0.1, f3_skip_channels=64, final_loss_weight=1.0, aux_f2_loss_weight=0.3, aux_f3_loss_weight=0.15):
        super().__init__()
        self.num_classes = 1
        self.visual_encoder, _ = get_segformer_encoder(vis_channels, segformer_id)
        self.vis_channels_list = [64, 128, 320, 512]
        self.air_encoder = AirQualityEncoder_ConvNeXt(in_channels=air_channels_input, channels_list=air_channels_list)

        self.film_f1 = FiLMLayer(air_dim=air_channels_list[0], vis_dim=self.vis_channels_list[0])
        self.gate_f1 = DynamicFusionGate(vis_dim=self.vis_channels_list[0], air_dim=air_channels_list[0])
        self.film_f2 = FiLMLayer(air_dim=air_channels_list[1], vis_dim=self.vis_channels_list[1])
        self.gate_f2 = DynamicFusionGate(vis_dim=self.vis_channels_list[1], air_dim=air_channels_list[1])

        self.ca_f3 = CrossAttentionBlock(vis_dim=self.vis_channels_list[2], air_dim=air_channels_list[2], num_heads=8, drop_path=drop_path_rate)
        self.ca_f4 = CrossAttentionBlock(vis_dim=self.vis_channels_list[3], air_dim=air_channels_list[3], num_heads=8, drop_path=drop_path_rate)

        self.decoder = ASPPDecoder(
            in_channels_high=self.vis_channels_list[3], in_channels_f3=self.vis_channels_list[2],  
            in_channels_low=self.vis_channels_list[1], in_channels_f1=self.vis_channels_list[0],
            num_classes=self.num_classes, f3_channels=f3_skip_channels, low_level_channels=64, f1_channels=64,
            decoder_channels=decoder_channels, num_decoder_blocks=num_decoder_blocks, drop_path_rate=drop_path_rate
        )

        self.loss_focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, ignore_index=255)
        self.loss_lov = LovaszLoss(ignore_index=255) 
        self.focal_weight = focal_weight
        self.lov_weight = lov_weight 
        self.final_loss_weight = final_loss_weight
        self.aux_f2_loss_weight = aux_f2_loss_weight
        self.aux_f3_loss_weight = aux_f3_loss_weight

    def _compute_loss(self, logits, labels):
        if logits.shape[-2:] != labels.shape[-2:]:
            upsampled_logits = F.interpolate(logits, size=labels.shape[-2:], mode='bilinear', align_corners=False)
        else:
            upsampled_logits = logits
        labels_float = labels.unsqueeze(1).float()
        loss_f = self.loss_focal(upsampled_logits, labels_float, labels)
        loss_l = self.loss_lov(upsampled_logits, labels_float) 
        return (self.focal_weight * loss_f) + (self.lov_weight * loss_l)

    def forward(self, pixel_values, air_values, labels=None):
        vis_features = self.visual_encoder(pixel_values, output_hidden_states=True, return_dict=True).hidden_states
        air_features = self.air_encoder(air_values)

        f_vis_f1, f_air_f1 = vis_features[0], air_features[0]
        fused_f1 = f_vis_f1 * (1.0 - self.gate_f1(f_vis_f1, f_air_f1)) + self.film_f1(f_vis_f1, f_air_f1) * self.gate_f1(f_vis_f1, f_air_f1)

        f_vis_f2, f_air_f2 = vis_features[1], air_features[1]
        fused_f2 = f_vis_f2 * (1.0 - self.gate_f2(f_vis_f2, f_air_f2)) + self.film_f2(f_vis_f2, f_air_f2) * self.gate_f2(f_vis_f2, f_air_f2)
        
        fused_f3 = self.ca_f3(vis_features[2], air_features[2])
        fused_f4 = self.ca_f4(vis_features[3], air_features[3])

        logits_dict = self.decoder([fused_f1, fused_f2, fused_f3, fused_f4])
        
        loss = None
        if labels is not None:
            loss_final = self._compute_loss(logits_dict["final"], labels)
            loss_aux_f2 = self._compute_loss(logits_dict["aux_f2"], labels)
            loss_aux_f3 = self._compute_loss(logits_dict["aux_f3"], labels)
            loss = (self.final_loss_weight * loss_final) + (self.aux_f2_loss_weight * loss_aux_f2) + (self.aux_f3_loss_weight * loss_aux_f3)
            
        upsampled_final_logits = logits_dict["final"]
        return {"loss": loss, "logits": upsampled_final_logits.squeeze(1)}
