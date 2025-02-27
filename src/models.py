import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


# ===== Segmentation model 01. UNet =====
class UNet(nn.Module):
    def __init__(self, in_channels, num_classes, pretrained=False):
        super(UNet, self).__init__()
        self.resnet = timm.create_model('resnet50', pretrained=pretrained, features_only=True, in_chans=in_channels)
        self.channels = self.resnet.feature_info.channels()

        decoder_channels = [256, 128, 64, 32]
        self.decoder = nn.ModuleList()
        self.reduce_channels = nn.ModuleList()

        for idx in range(len(decoder_channels)):
            if idx == 0:
                in_ch = self.channels[-1]
            else:
                in_ch = decoder_channels[idx - 1]

            skip_ch = self.channels[-(idx + 2)] if idx < len(self.channels) - 1 else 0
            out_ch = decoder_channels[idx]

            if skip_ch > 0:
                self.reduce_channels.append(
                    nn.Sequential(
                        nn.Conv2d(skip_ch, skip_ch // 2, 1),
                        nn.BatchNorm2d(num_features=skip_ch // 2),
                        nn.ReLU(inplace=True)
                    )
                )

            self.decoder.append(
                nn.Sequential(
                    nn.Conv2d(in_ch + (skip_ch // 2 if skip_ch > 0 else 0), out_ch, 3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_ch, out_ch, 3, padding=1),
                    nn.BatchNorm2d(num_features=out_ch),
                    nn.ReLU(inplace=True),
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
                )
            )

        self.final_conv = nn.Conv2d(decoder_channels[-1], num_classes, kernel_size=1)

    def forward(self, x):
        # execute encoder
        features = self.resnet(x)

        # execute decoder
        x = features[-1]

        for idx, decoder_block in enumerate(self.decoder):
            if idx < len(features) - 1:
                skip = features[-(idx + 2)]
                x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
                skip = self.reduce_channels[idx](skip)
                x = torch.cat([x, skip], dim=1)

            x = decoder_block(x)

        x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
        x = self.final_conv(x)

        return x


# ===== Segmentation model 02. Nested UNet =====
class NestedUNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=4, pretrained=False):
        super().__init__()
        self.backbone = timm.create_model('resnet50',
                                          pretrained=pretrained,
                                          features_only=True,
                                          in_chans=in_channels)

        # ResNet50 output channels: [256, 512, 1024, 2048]
        self.channels = self.backbone.feature_info.channels()

        # First level - use first backbone feature
        self.conv0_0 = nn.Sequential(
            nn.Conv2d(self.channels[0], 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.conv1_0 = nn.Sequential(
            nn.Conv2d(self.channels[1], 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self.conv2_0 = nn.Sequential(
            nn.Conv2d(self.channels[2], 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self.conv3_0 = nn.Sequential(
            nn.Conv2d(self.channels[3], 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        # For x0_1: cat(x0_0[256], up(x1_0)[512]) = 768 channels
        self.conv0_1 = nn.Sequential(
            nn.Conv2d(768, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # For x1_1: cat(x1_0[512], up(x2_0)[512]) = 1024 channels
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(1024, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        # For x2_1: cat(x2_0[512], up(x3_0)[512]) = 1024 channels
        self.conv2_1 = nn.Sequential(
            nn.Conv2d(1024, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        # For x0_2: cat(x0_0[256], x0_1[256], up(x1_1)[512]) = 1024 channels
        self.conv0_2 = nn.Sequential(
            nn.Conv2d(1024, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # For x1_2: cat(x1_0[512], x1_1[512], up(x2_1)[512]) = 1536 channels
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(1536, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        # For x0_3: cat(x0_0[256], x0_1[256], x0_2[256], up(x1_2)[512]) = 1280 channels
        self.conv0_3 = nn.Sequential(
            nn.Conv2d(1280, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # For x0_4: cat(x0_0[256], x0_1[256], x0_2[256], x0_3[256], up(x1_3)[512]) = 1536 channels
        self.conv0_4 = nn.Sequential(
            nn.Conv2d(1536, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.final1 = nn.Conv2d(256, num_classes, kernel_size=1)
        self.final2 = nn.Conv2d(256, num_classes, kernel_size=1)
        self.final3 = nn.Conv2d(256, num_classes, kernel_size=1)
        self.final4 = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        features = list(self.backbone(x))

        x0_0 = self.conv0_0(features[0])
        x1_0 = self.conv1_0(features[1])
        x2_0 = self.conv2_0(features[2])
        x3_0 = self.conv3_0(features[3])

        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))

        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))

        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_2)], 1))

        x0_1_up = F.interpolate(x0_1, size=(256, 256), mode='bilinear', align_corners=True)
        x0_2_up = F.interpolate(x0_2, size=(256, 256), mode='bilinear', align_corners=True)
        x0_3_up = F.interpolate(x0_3, size=(256, 256), mode='bilinear', align_corners=True)
        x0_4_up = F.interpolate(x0_4, size=(256, 256), mode='bilinear', align_corners=True)

        if self.training:
            return [
                self.final1(x0_1_up),
                self.final2(x0_2_up),
                self.final3(x0_3_up),
                self.final4(x0_4_up)
            ]
        else:
            return self.final4(x0_4_up)


# ===== Segmentation model 03. SwinUNet =====
class SwinUNet(nn.Module):
    def __init__(self, in_channels, num_classes, pretrained=False):
        super(SwinUNet, self).__init__()
        # set backbone
        self.backbone = timm.create_model('swin_base_patch4_window7_224', pretrained=pretrained, features_only=True, in_chans=in_channels)
        self.channels = self.backbone.feature_info.channels()
        backbone_out_channels = self.channels[-1]

        # set decoder
        self.decoder1 = nn.Sequential(
            nn.Conv2d(backbone_out_channels, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True)
        )

        self.decoder2 = nn.Sequential(
            nn.Conv2d(256 + self.channels[-2], 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True)
        )

        self.decoder3 = nn.Sequential(
            nn.Conv2d(128 + self.channels[-3], 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True)
        )

        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # interpolate input to 224x224
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)

        # get features from backbone
        features = self.backbone(x)
        features = [feature.permute(0, 3, 1, 2) for feature in features] # BHWC -> BCHW

        # execute decoder
        x = features[-1]
        x = self.decoder1(x)
        x = F.interpolate(x, size=features[-2].shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, features[-2]], dim=1)

        x = self.decoder2(x)
        x = F.interpolate(x, size=features[-3].shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, features[-3]], dim=1)

        x = self.decoder3(x)

        x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
        x = self.final_conv(x)

        return x


# ===== Segmentation model 04. TransUNet =====
class TransUNet(nn.Module):
    def __init__(self, in_channels, num_classes, pretrained=False):
        super(TransUNet, self).__init__()
        self.resnet = timm.create_model('resnet50', pretrained=pretrained, features_only=True, in_chans=in_channels)
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=pretrained, in_chans=in_channels)

        self.channels = self.resnet.feature_info.channels()
        self.vit_dim = self.vit.embed_dim

        decoder_channels = [256, 128, 64, 32]
        vit_proj_dims = [512, 256]
        self.decoder = nn.ModuleList()
        self.reduce_channels = nn.ModuleList()

        self.vit_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.vit_dim, ch),
                nn.ReLU(inplace=True)
            ) for ch in vit_proj_dims
        ])

        self.vit_reduce = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(vit_proj_dims[i], vit_proj_dims[i] // 4, 1),
                nn.BatchNorm2d(vit_proj_dims[i] // 4),
                nn.ReLU(inplace=True)
            ) for i in range(2)
        ])

        for idx in range(len(decoder_channels)):
            if idx == 0:
                in_ch = self.channels[-1]
                vit_ch = vit_proj_dims[0] // 4 if idx < 2 else 0
            else:
                in_ch = decoder_channels[idx - 1]
                vit_ch = vit_proj_dims[idx] // 4 if idx < 2 else 0

            skip_ch = self.channels[-(idx + 2)] if idx < len(self.channels) - 1 else 0
            out_ch = decoder_channels[idx]

            if skip_ch > 0:
                self.reduce_channels.append(
                    nn.Sequential(
                        nn.Conv2d(skip_ch, skip_ch // 2, 1),
                        nn.BatchNorm2d(num_features=skip_ch // 2),
                        nn.ReLU(inplace=True)
                    )
                )

            total_ch = in_ch + (skip_ch // 2 if skip_ch > 0 else 0) + vit_ch

            self.decoder.append(
                nn.Sequential(
                    nn.Conv2d(total_ch, out_ch, 3, padding=1),
                    nn.BatchNorm2d(num_features=out_ch),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_ch, out_ch, 3, padding=1),
                    nn.BatchNorm2d(num_features=out_ch),
                    nn.ReLU(inplace=True),
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
                )
            )

        self.final_conv = nn.Conv2d(decoder_channels[-1], num_classes, kernel_size=1)

    def _process_vit(self, x):
        B = x.shape[0]
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)

        x = self.vit.patch_embed(x)

        cls_token = self.vit.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        x = self.vit.pos_drop(x + self.vit.pos_embed)
        x = self.vit.blocks(x)
        x = self.vit.norm(x)

        return x[:, 1:]

    def _process_vit_features(self, vit_features, idx, h, w):
        B = vit_features.shape[0]
        x = self.vit_projections[idx](vit_features)

        x = x.transpose(1, 2).reshape(B, -1, 14, 14)

        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)

        x = self.vit_reduce[idx](x)

        return x

    def forward(self, x):
        # get CNN features
        features = self.resnet(x)
        # get ViT features
        vit_features = self._process_vit(x)

        # execute decoder
        x = features[-1]

        for idx, decoder_block in enumerate(self.decoder):
            if idx < len(features) - 1:
                skip = features[-(idx + 2)]
                x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
                skip = self.reduce_channels[idx](skip)
                x = torch.cat([x, skip], dim=1)

            if idx < 2:
                vit_feat = self._process_vit_features(vit_features, idx, x.shape[2], x.shape[3])
                x = torch.cat([x, vit_feat], dim=1)

            x = decoder_block(x)

        x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
        x = self.final_conv(x)

        return x


# ===== Siamese model 01. BiUNet =====
class BiUNet(nn.Module):
    def __init__(self, in_channels, num_classes, pretrained=False):
        super(BiUNet, self).__init__()
        self.resnet = timm.create_model('resnet50', pretrained=pretrained, features_only=True, in_chans=in_channels)
        self.channels = self.resnet.feature_info.channels()

        decoder_channels = [256, 128, 64, 32]
        self.decoder = nn.ModuleList()
        self.reduce_channels = nn.ModuleList()

        for idx in range(len(decoder_channels)):
            if idx == 0:
                in_ch = self.channels[-1] * 2
            else:
                in_ch = decoder_channels[idx - 1]

            skip_ch = self.channels[-(idx + 2)] if idx < len(self.channels) - 1 else 0
            out_ch = decoder_channels[idx]

            if skip_ch > 0:
                self.reduce_channels.append(
                    nn.Sequential(
                        nn.Conv2d(skip_ch, skip_ch // 4, 1),
                        nn.BatchNorm2d(num_features=skip_ch // 4),
                        nn.ReLU(inplace=True)
                    )
                )

            self.decoder.append(
                nn.Sequential(
                    nn.Conv2d(in_ch + (skip_ch // 2 if skip_ch > 0 else 0), out_ch, 3, padding=1),
                    nn.BatchNorm2d(num_features=out_ch),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_ch, out_ch, 3, padding=1),
                    nn.BatchNorm2d(num_features=out_ch),
                    nn.ReLU(inplace=True),
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
                )
            )

        self.final_conv = nn.Conv2d(decoder_channels[-1], num_classes, kernel_size=1)

    def forward(self, x1, x2):
        features_x1 = self.resnet(x1)
        features_x2 = self.resnet(x2)

        x = torch.cat([features_x1[-1], features_x2[-1]], dim=1)

        for idx, decoder_block in enumerate(self.decoder):
            if idx < len(features_x1) - 1:
                skip_x1 = features_x1[-(idx + 2)]
                skip_x2 = features_x2[-(idx + 2)]

                x = F.interpolate(x, size=skip_x1.shape[-2:], mode='bilinear', align_corners=False)

                skip_x1 = self.reduce_channels[idx](skip_x1)
                skip_x2 = self.reduce_channels[idx](skip_x2)

                x = torch.cat([x, skip_x1, skip_x2], dim=1)

            x = decoder_block(x)

        x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
        x = self.final_conv(x)

        return x


# ===== Siamese model 02. SNUNet =====
class SNUNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=4, pretrained=False):
        super().__init__()
        self.backbone = timm.create_model('vgg16_bn',
                                          pretrained=pretrained,
                                          features_only=True,
                                          out_indices=(1, 2, 3, 4, 5),
                                          in_chans=in_channels)

        self.channels = [128, 256, 512, 512, 512]
        nb_filter = self.channels

        self.proj1 = nn.Conv2d(256, 128, 1)
        self.proj2 = nn.Conv2d(512, 256, 1)
        self.proj3 = nn.Conv2d(512, 512, 1)
        self.proj4 = nn.Conv2d(512, 512, 1)

        self.conv0_1 = nn.Sequential(
            nn.Conv2d(nb_filter[0] * 3, nb_filter[0], 3, padding=1),
            nn.BatchNorm2d(nb_filter[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(nb_filter[0], nb_filter[0], 3, padding=1),
            nn.BatchNorm2d(nb_filter[0]),
            nn.ReLU(inplace=True)
        )

        self.conv1_1 = nn.Sequential(
            nn.Conv2d(nb_filter[1] * 3, nb_filter[1], 3, padding=1),
            nn.BatchNorm2d(nb_filter[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(nb_filter[1], nb_filter[1], 3, padding=1),
            nn.BatchNorm2d(nb_filter[1]),
            nn.ReLU(inplace=True)
        )

        self.conv2_1 = nn.Sequential(
            nn.Conv2d(nb_filter[2] * 3, nb_filter[2], 3, padding=1),
            nn.BatchNorm2d(nb_filter[2]),
            nn.ReLU(inplace=True),
            nn.Conv2d(nb_filter[2], nb_filter[2], 3, padding=1),
            nn.BatchNorm2d(nb_filter[2]),
            nn.ReLU(inplace=True)
        )

        self.conv3_1 = nn.Sequential(
            nn.Conv2d(nb_filter[3] * 3, nb_filter[3], 3, padding=1),
            nn.BatchNorm2d(nb_filter[3]),
            nn.ReLU(inplace=True),
            nn.Conv2d(nb_filter[3], nb_filter[3], 3, padding=1),
            nn.BatchNorm2d(nb_filter[3]),
            nn.ReLU(inplace=True)
        )

        self.conv0_2 = nn.Sequential(
            nn.Conv2d(nb_filter[0] * 4, nb_filter[0], 3, padding=1),
            nn.BatchNorm2d(nb_filter[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(nb_filter[0], nb_filter[0], 3, padding=1),
            nn.BatchNorm2d(nb_filter[0]),
            nn.ReLU(inplace=True)
        )

        self.conv1_2 = nn.Sequential(
            nn.Conv2d(nb_filter[1] * 4, nb_filter[1], 3, padding=1),
            nn.BatchNorm2d(nb_filter[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(nb_filter[1], nb_filter[1], 3, padding=1),
            nn.BatchNorm2d(nb_filter[1]),
            nn.ReLU(inplace=True)
        )

        self.conv2_2 = nn.Sequential(
            nn.Conv2d(nb_filter[2] * 4, nb_filter[2], 3, padding=1),
            nn.BatchNorm2d(nb_filter[2]),
            nn.ReLU(inplace=True),
            nn.Conv2d(nb_filter[2], nb_filter[2], 3, padding=1),
            nn.BatchNorm2d(nb_filter[2]),
            nn.ReLU(inplace=True)
        )

        self.conv0_3 = nn.Sequential(
            nn.Conv2d(nb_filter[0] * 5, nb_filter[0], 3, padding=1),
            nn.BatchNorm2d(nb_filter[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(nb_filter[0], nb_filter[0], 3, padding=1),
            nn.BatchNorm2d(nb_filter[0]),
            nn.ReLU(inplace=True)
        )

        self.conv1_3 = nn.Sequential(
            nn.Conv2d(nb_filter[1] * 5, nb_filter[1], 3, padding=1),
            nn.BatchNorm2d(nb_filter[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(nb_filter[1], nb_filter[1], 3, padding=1),
            nn.BatchNorm2d(nb_filter[1]),
            nn.ReLU(inplace=True)
        )

        self.conv0_4 = nn.Sequential(
            nn.Conv2d(nb_filter[0] * 6, nb_filter[0], 3, padding=1),
            nn.BatchNorm2d(nb_filter[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(nb_filter[0], nb_filter[0], 3, padding=1),
            nn.BatchNorm2d(nb_filter[0]),
            nn.ReLU(inplace=True)
        )

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, x1, x2):
        features1 = list(self.backbone(x1))
        features2 = list(self.backbone(x2))

        [x0_0_1, x1_0_1, x2_0_1, x3_0_1, x4_0_1] = features1
        [x0_0_2, x1_0_2, x2_0_2, x3_0_2, x4_0_2] = features2

        x0_0 = torch.abs(x0_0_1 - x0_0_2)
        x1_0 = torch.abs(x1_0_1 - x1_0_2)
        x2_0 = torch.abs(x2_0_1 - x2_0_2)
        x3_0 = torch.abs(x3_0_1 - x3_0_2)
        x4_0 = torch.abs(x4_0_1 - x4_0_2)

        up_x1_0 = self.proj1(self.up(x1_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, up_x1_0, x0_0], 1))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.proj2(self.up(x2_0)), x1_0], 1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.proj3(self.up(x3_0)), x2_0], 1))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.proj4(self.up(x4_0)), x3_0], 1))

        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(self.proj1(x1_1)), x0_0], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(self.proj2(x2_1)), x1_0], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(self.proj3(x3_1)), x2_0], 1))

        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(self.proj1(x1_2)), x0_0], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(self.proj2(x2_2)), x1_0], 1))

        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(self.proj1(x1_3)), x0_0], 1))

        # Upsampling before final convolutions
        x0_1_up = F.interpolate(x0_1, size=(256, 256), mode='bilinear', align_corners=True)
        x0_2_up = F.interpolate(x0_2, size=(256, 256), mode='bilinear', align_corners=True)
        x0_3_up = F.interpolate(x0_3, size=(256, 256), mode='bilinear', align_corners=True)
        x0_4_up = F.interpolate(x0_4, size=(256, 256), mode='bilinear', align_corners=True)

        if self.training:
            return [
                self.final1(x0_1_up),
                self.final2(x0_2_up),
                self.final3(x0_3_up),
                self.final4(x0_4_up)
            ]
        else:
            return self.final4(x0_4_up)


# ===== Siamese model 03. ChangeFormer =====
class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x1, x2):
        B, N, C = x1.shape

        q = self.q(x1).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k(x2).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v(x2).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class BidirectionalTransformer(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1_1 = nn.LayerNorm(dim)
        self.norm1_2 = nn.LayerNorm(dim)
        self.norm2_1 = nn.LayerNorm(dim)
        self.norm2_2 = nn.LayerNorm(dim)

        self.attn1to2 = CrossAttention(dim, num_heads, qkv_bias, attn_drop, drop)
        self.attn2to1 = CrossAttention(dim, num_heads, qkv_bias, attn_drop, drop)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp1 = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )

    def forward(self, x1, x2):
        # Cross attention
        attn1 = self.attn1to2(self.norm1_1(x1), self.norm1_2(x2))
        x1 = x1 + attn1
        x1 = x1 + self.mlp1(self.norm2_1(x1))

        attn2 = self.attn2to1(self.norm1_1(x2), self.norm1_2(x1))
        x2 = x2 + attn2
        x2 = x2 + self.mlp2(self.norm2_2(x2))

        return x1, x2


class ChangeFormer(nn.Module):
    def __init__(self, in_channels, num_classes, pretrained=False):
        super().__init__()
        # swin transformer backbone
        self.backbone = timm.create_model(
            'swin_base_patch4_window7_224',
            pretrained=pretrained,
            features_only=True,
            in_chans=in_channels
        )
        self.channels = self.backbone.feature_info.channels()

        # bidirectional transformer blocks for each scale
        self.transformers = nn.ModuleList([
            BidirectionalTransformer(
                dim=ch,
                num_heads=ch // 32 if ch >= 32 else 1,
                mlp_ratio=4,
                qkv_bias=True,
                drop=0.1,
                attn_drop=0.1
            ) for ch in self.channels
        ])

        # feature refinement
        self.refine_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch * 2, ch, 3, padding=1),
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True)
            ) for ch in self.channels
        ])

        # decoder
        decoder_channels = [256, 128, 64, 32]
        self.decoder = nn.ModuleList()

        for idx in range(len(decoder_channels)):
            if idx == 0:
                in_ch = self.channels[-1]
            else:
                in_ch = decoder_channels[idx - 1]

            skip_ch = self.channels[-(idx + 2)] if idx < len(self.channels) - 1 else 0
            out_ch = decoder_channels[idx]

            self.decoder.append(nn.Sequential(
                nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            ))

        # Final classifier
        self.final_conv = nn.Sequential(
            nn.Conv2d(decoder_channels[-1], decoder_channels[-1], 3, padding=1),
            nn.BatchNorm2d(decoder_channels[-1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_channels[-1], num_classes, 1)
        )

    def _process_features(self, x):
        B, C, H, W = x.shape
        return x.flatten(2).transpose(1, 2), H, W

    def forward(self, x1, x2):
        # extract features
        x1 = F.interpolate(x1, size=(224, 224), mode='bilinear', align_corners=False)
        x2 = F.interpolate(x2, size=(224, 224), mode='bilinear', align_corners=False)

        feats_x1 = self.backbone(x1)
        feats_x2 = self.backbone(x2)
        feats_x1 = [f.permute(0, 3, 1, 2) for f in feats_x1]  # BHWC -> BCHW
        feats_x2 = [f.permute(0, 3, 1, 2) for f in feats_x2]  # BHWC -> BCHW

        # process features with bidirectional transformers
        change_feats = []
        for feat1, feat2, transformer, refine in zip(feats_x1, feats_x2, self.transformers, self.refine_convs):
            # convert to sequence
            feat1_seq, H, W = self._process_features(feat1)
            feat2_seq, _, _ = self._process_features(feat2)

            # apply transformer
            feat1_attn, feat2_attn = transformer(feat1_seq, feat2_seq)

            # reshape back to spatial features
            feat1_attn = feat1_attn.transpose(1, 2).reshape(-1, feat1.size(1), H, W)
            feat2_attn = feat2_attn.transpose(1, 2).reshape(-1, feat2.size(1), H, W)

            # compute change features with refinement
            change_feat = torch.cat([feat1_attn, feat2_attn], dim=1)
            change_feat = refine(change_feat)
            change_feats.append(change_feat)

        # decoder with skip connections
        x = change_feats[-1]
        for idx, decoder_block in enumerate(self.decoder):
            if idx < len(change_feats) - 1:
                skip = change_feats[-(idx + 2)]
                x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
                x = torch.cat([x, skip], dim=1)
            x = decoder_block(x)

        # final prediction
        x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
        x = self.final_conv(x)

        return x


# ===== Siamese model 04. TransFireNet =====
class TransFireNet(nn.Module):
    def __init__(self, in_channels, num_classes, pretrained=False):
        super(TransFireNet, self).__init__()
        self.pvt = timm.create_model('pvt_v2_b2', pretrained=pretrained, features_only=True, in_chans=in_channels)
        self.channels = self.pvt.feature_info.channels()

        decoder_channels = [256, 128, 64, 32]
        self.decoder = nn.ModuleList()

        for idx in range(len(decoder_channels)):
            in_ch = self.channels[-1] if idx == 0 else decoder_channels[idx - 1]
            skip_ch = self.channels[-(idx + 2)] if idx < len(self.channels) - 1 else 0
            out_ch = decoder_channels[idx]

            total_ch = in_ch + skip_ch * 2 if skip_ch > 0 else in_ch

            self.decoder.append(
                nn.Sequential(
                    nn.Conv2d(total_ch, out_ch, 3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_ch, out_ch, 3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
                )
            )

        self.final = nn.Sequential(
            nn.Conv2d(decoder_channels[-1], decoder_channels[-1], 3, padding=1),
            nn.BatchNorm2d(decoder_channels[-1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_channels[-1], num_classes, 1)
        )

    def forward(self, x1, x2):
        feats_x1 = self.pvt(x1)
        feats_x2 = self.pvt(x2)

        diff_feats = []
        for f1, f2 in zip(feats_x1, feats_x2):
            diff = torch.abs(f1 - f2)
            diff_feats.append(diff)

        x = diff_feats[-1]

        for idx, decoder_block in enumerate(self.decoder):
            if idx < len(diff_feats) - 1:
                skip_diff = diff_feats[-(idx + 2)]
                skip_orig = feats_x2[-(idx + 2)]

                x = F.interpolate(x, size=skip_diff.shape[-2:], mode='bilinear', align_corners=False)
                x = torch.cat([x, skip_diff, skip_orig], dim=1)

            x = decoder_block(x)

        x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
        x = self.final(x)

        return x


if __name__ == "__main__":
    in_channels = 3
    num_classes = 4
    batch_size = 2
    input_size = 256

    ## ===== Siamese network test =====
    x1 = torch.randn(batch_size, in_channels, input_size, input_size)
    x2 = torch.randn(batch_size, in_channels, input_size, input_size)

    print(f"BiUNet Input shape: {x1.shape}, {x2.shape}")
    biunet = BiUNet(in_channels, num_classes, pretrained=False)
    biunet.eval()
    with torch.no_grad():
        out_biunet = biunet(x1, x2)
    print(f"BiUNet Output shape: {out_biunet.shape}")

    print(f"SNUNet Input shape: {x1.shape}, {x2.shape}")
    snunet = SNUNet(in_channels, num_classes, pretrained=False)
    snunet.eval()
    with torch.no_grad():
        out_snunet = snunet(x1, x2)
    print(f"SNUNet Output shape: {out_snunet.shape}")

    print(f"ChangeFormer Input shape: {x1.shape}, {x2.shape}")
    changeformer = ChangeFormer(in_channels, num_classes, pretrained=False)
    changeformer.eval()
    with torch.no_grad():
        out_changeformer = changeformer(x1, x2)
    print(f"ChangeFormer Output shape: {out_changeformer.shape}")

    print(f"TransFireNet Input shape: {x1.shape}, {x2.shape}")
    transfirenet = TransFireNet(in_channels, num_classes, pretrained=False)
    transfirenet.eval()
    with torch.no_grad():
        out_transfirenet = transfirenet(x1, x2)
    print(f"TransFireNet Output shape: {out_transfirenet.shape}")

    ## ===== Segmentation test =====
    x = torch.randn(batch_size, in_channels * 2, input_size, input_size)

    print(f"UNet Input shape: {x.shape}")
    unet = UNet(in_channels * 2, num_classes, pretrained=False)
    unet.eval()
    with torch.no_grad():
        out_unet = unet(x)
    print(f"UNet Output shape: {out_unet.shape}")

    print(f"NestedUNet Input shape: {x.shape}")
    nestedunet = NestedUNet(in_channels * 2, num_classes, pretrained=False)
    nestedunet.eval()
    with torch.no_grad():
        out_nestedunet = nestedunet(x)
    print(f"NestedUNet Output shape: {out_nestedunet.shape}")

    print(f"SwinUNet Input shape: {x.shape}")
    swinunet = SwinUNet(in_channels * 2, num_classes, pretrained=False)
    swinunet.eval()
    with torch.no_grad():
        out_swinunet = swinunet(x)
    print(f"SwinUNet Output shape: {out_swinunet.shape}")

    print(f"TransUNet Input shape: {x.shape}")
    transunet = TransUNet(in_channels * 2, num_classes, pretrained=False)
    transunet.eval()
    with torch.no_grad():
        out_transunet = transunet(x)
    print(f"TransUNet Output shape: {out_transunet.shape}")