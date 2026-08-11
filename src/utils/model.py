import torch
from torch import nn
from torch.nn.functional import normalize, interpolate
from torchvision.models.resnet import Bottleneck, BasicBlock, conv1x1
from peft import LoraConfig, get_peft_model, TaskType
import torch.nn.functional as F
############################################
# BackBoneModel 
############################################
class BackBoneModel(nn.Module):
    def __init__(self,
                 input_shape: int,
                 output_shape: int,
                 input_size: int = 64):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=128,
                      kernel_size=7,
                      stride=2,
                      padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(in_channels=128,
                      out_channels=128,
                      kernel_size=7,
                      stride=2,
                      padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=128,
                      out_channels=128,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(in_channels=128,
                      out_channels=128,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2)
        )

        with torch.no_grad():
            dummy = torch.zeros(1, input_shape, input_size, input_size)
            features = self.conv_block_1(dummy)
            in_features = features.view(1, -1).size(1)
        
        self.linear_layer = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.0, inplace=True),
            nn.Linear(in_features, output_shape),
        )

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.linear_layer(x)
        return x


############################################
# Cross-Attention 
############################################
class CrossModalFusion(nn.Module):
    def __init__(self, vis_dim, text_dim, embed_dim, num_heads=4):
        """
        Args:
            vis_dim: 视觉骨干网络的输出维度
            text_dim: DNA LLM 的输出维度
            embed_dim: 融合后的特征维度
        """
        super().__init__()
        
        # 1. 投影层
        self.vis_proj = nn.Linear(vis_dim, embed_dim)
        self.text_proj = nn.Linear(text_dim, embed_dim)
        
        # 2. Cross-Attention (Query=Vision, Key/Value=Text)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        
        # 3. Feed Forward & Norm
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )

    def forward(self, vis_feat, text_feat, text_mask=None):
        """
        vis_feat: (B, VisDim)
        text_feat: (B, SeqLen, TextDim)
        text_mask: (B, SeqLen)
        """
        # --- 维度对齐 ---
        query = self.vis_proj(vis_feat).unsqueeze(1) # (B, 1, E)
        key = self.text_proj(text_feat) # (B, L, E)
        value = key 
        
        # --- Mask 处理 ---
        # 如果 text_mask 是 1=Valid, 0=Pad (HuggingFace 默认)，则 PyTorch Attention 需要取反
        # PyTorch: True/1 表示被忽略
        key_padding_mask = (text_mask == 0) if text_mask is not None else None
        
        # --- Cross Attention ---
        attn_output, _ = self.cross_attn(query, key, value, key_padding_mask=key_padding_mask)
        
        # Residual + Norm
        x = self.norm1(query + attn_output)
        
        # FFN + Residual + Norm
        x2 = self.ffn(x)
        x = self.norm2(x + x2)
        
        return x.squeeze(1)





import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model


class Gate1(nn.Module):
    def __init__(self, x1_dim, x2_dim):
        """
        x1_dim: 视觉特征维度 (Backbone output)
        x2_dim: 文本特征维度 (LLM hidden size)
        """
        super().__init__()
        # 将文本投影到视觉维度
        self.proj = nn.Linear(x2_dim, x1_dim)
        # 计算门控系数的线性层
        self.wz = nn.Linear(x1_dim * 2, x1_dim)
    
    def forward(self, x1, x2):

        x2_proj = self.proj(x2)
        
     
        gate_lambda = self.wz(torch.cat([x1, x2_proj], dim=1)).sigmoid()
        if self.training and not hasattr(self, '_logged_gate_debug'):
            print(f"\n[DEBUG Fusion] Gate Visual: {(1-gate_lambda).mean().item():.3f} | Gate Text: {gate_lambda.mean().item():.3f}")
            self._logged_gate_debug = True
  
        out = (1 - gate_lambda) * x1 + gate_lambda * x2_proj
        
        return out


def normalize(x, dim=1):
    return F.normalize(x, p=2, dim=dim)

class Network(nn.Module):
    def __init__(self, backbone, dna_llm, rep_dim, feature_dim, class_num, fusion_dim=512, use_llm=False, fusion_type='gate'):
        super(Network, self).__init__()
        
        self.use_llm = use_llm
        self.backbone = backbone
        self.vis_dim = rep_dim
        self.feature_dim = feature_dim
        self.cluster_num = class_num

        if self.use_llm:
          
            if dna_llm is None:
                raise ValueError("use_llm=True, but dna_llm is None!")
                
            self.text_encoder = dna_llm
            self.text_dim = dna_llm.config.hidden_size 
            
          
            for param in self.text_encoder.parameters():
                param.requires_grad = False

            peft_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                inference_mode=False, 
                r=8,
                lora_alpha=32,
                lora_dropout=0.1,
                target_modules=["query", "value","key"] 
            )
            
            self.text_encoder = get_peft_model(self.text_encoder, peft_config)
            
            # Fusion selection
            self.fusion_type = fusion_type
            self.fusion_dim = fusion_dim
            if self.fusion_type == 'gate':
         
                self.fusion = Gate1(x1_dim=self.vis_dim, x2_dim=self.text_dim)
                projector_input_dim = self.vis_dim
            elif self.fusion_type == 'cross_attn':
                self.fusion = CrossModalFusion(self.vis_dim, self.text_dim, fusion_dim)
                projector_input_dim = fusion_dim
            else:
                raise ValueError(f"Unknown fusion_type: {self.fusion_type}")
            
        else:
      
            self.text_encoder = None
            self.fusion = None
            projector_input_dim = self.vis_dim

    
        self.instance_projector = nn.Sequential(
            nn.Linear(projector_input_dim, projector_input_dim),
            nn.ReLU(),
            nn.Linear(projector_input_dim, self.feature_dim),
        )
        
        self.cluster_projector = nn.Sequential(
            nn.Linear(projector_input_dim, projector_input_dim),
            nn.ReLU(),
            nn.Linear(projector_input_dim, self.cluster_num),
            nn.Softmax(dim=1)
        )

    def _mean_pooling(self, token_embeddings, attention_mask):
   
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        
    
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        
    
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
  
        return sum_embeddings / sum_mask

    def forward(self, x_i, x_j, input_ids_weak=None, mask_weak=None, input_ids_strong=None, mask_strong=None, run_visual_only=False):
 
        h_i = self.backbone(x_i) # [Batch, Vis_Dim]
        h_j = self.backbone(x_j)
        if self.training and self.use_llm and (not run_visual_only):
      
            drop_visual_prob = 0.3
            
   
            if torch.rand(1).item() < drop_visual_prob:
                h_i = torch.zeros_like(h_i) 
            
     
            if torch.rand(1).item() < drop_visual_prob:
                h_j = torch.zeros_like(h_j)
        # ============================================================


        should_fuse = self.use_llm and (not run_visual_only)
        

        if should_fuse:
     
            input_ids_i, mask_i = input_ids_weak, mask_weak
            input_ids_j, mask_j = input_ids_strong, mask_strong

            # --- View i (Weak) ---
            out_i = self.text_encoder(input_ids=input_ids_i, attention_mask=mask_i)
            full_text_feat_i = out_i.last_hidden_state if hasattr(out_i, 'last_hidden_state') else out_i

            if self.fusion_type == 'gate':
            
                text_feat_i = self._mean_pooling(full_text_feat_i, mask_i)
                feat_i = self.fusion(h_i, text_feat_i)
               
            elif self.fusion_type == 'cross_attn':
                feat_i = self.fusion(h_i, full_text_feat_i, text_mask=mask_i)
            else:
                raise ValueError(f"Unknown fusion_type: {self.fusion_type}")

            # --- View j (Strong) ---
            out_j = self.text_encoder(input_ids=input_ids_j, attention_mask=mask_j)
            full_text_feat_j = out_j.last_hidden_state if hasattr(out_j, 'last_hidden_state') else out_j

            if self.fusion_type == 'gate':
               
                text_feat_j = self._mean_pooling(full_text_feat_j, mask_j)
                feat_j = self.fusion(h_j, text_feat_j)
            elif self.fusion_type == 'cross_attn':
                feat_j = self.fusion(h_j, full_text_feat_j, text_mask=mask_j)
            else:
                raise ValueError(f"Unknown fusion_type: {self.fusion_type}")

        else:
        
            feat_i = h_i
            feat_j = h_j

     
        z_i = normalize(self.instance_projector(feat_i), dim=1)
        z_j = normalize(self.instance_projector(feat_j), dim=1)

        c_i = self.cluster_projector(feat_i)
        c_j = self.cluster_projector(feat_j)

        return z_i, z_j, c_i, c_j

    def forward_cluster(self, x_img, input_ids=None, attention_mask=None, run_visual_only=False):
     
        h_vis = self.backbone(x_img)
        
      
        should_fuse = self.use_llm and (input_ids is not None) and (not run_visual_only)

        if should_fuse:
         
            out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
            full_text_feat = out.last_hidden_state if hasattr(out, 'last_hidden_state') else out

            if self.fusion_type == 'gate':
            
                h_text = self._mean_pooling(full_text_feat, attention_mask)
                feat = self.fusion(h_vis, h_text)
            elif self.fusion_type == 'cross_attn':
                feat = self.fusion(h_vis, full_text_feat, text_mask=attention_mask)
            else:
                raise ValueError(f"Unknown fusion_type: {self.fusion_type}")
        else:
           
            feat = h_vis
            
   
        c = self.cluster_projector(feat)
        return c

############################################
# ResNet & Utilities 
############################################

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.rep_dim = 512 * block.expansion

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def get_resnet(name):
    resnet18 = ResNet(block=BasicBlock, layers=[2, 2, 2, 2])
    resnet34 = ResNet(block=BasicBlock, layers=[3, 4, 6, 3])
    resnet50 = ResNet(block=Bottleneck, layers=[3, 4, 6, 3])

    resnets = {
        "ResNet18": resnet18,
        "ResNet34": resnet34,
        "ResNet50": resnet50,
    }
    if name not in resnets.keys():
        raise KeyError(f"{name} is not a valid ResNet version")
    return resnets[name]

