import torch
# print("Please define your ResNet in this file.")


class bottleneck_block(torch.nn.Module):
    def __init__(self, x_channels, first_filters=64, strides=1) -> None:
        super(bottleneck_block, self).__init__()

        self.direct_layers = torch.nn.Sequential(
            torch.nn.Conv2d(x_channels, first_filters,
                            kernel_size=(1, 1), stride=strides),
            # Batch Normalization
            torch.nn.BatchNorm2d(first_filters, eps=1e-05,
                                 momentum=0.1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(first_filters, first_filters,
                            kernel_size=(3, 3), stride=1, padding=1),
            # Batch Normalization
            torch.nn.BatchNorm2d(first_filters, eps=1e-05,
                                 momentum=0.1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(first_filters, first_filters*4,
                            kernel_size=(1, 1), stride=1),
        )

        self.identity_layer = torch.nn.Sequential(torch.nn.Conv2d(
            x_channels, first_filters*4, kernel_size=1, stride=1))

        if strides != 1:
            self.identity_layer = torch.nn.Conv2d(
                x_channels, first_filters*4, kernel_size=1, stride=strides)

        self.relu = torch.nn.ReLU()

    def forward(self, x):
        identity = self.identity_layer(x)
        x = self.direct_layers(x)

        x += identity

        x = self.relu(x)
        return x


class basic_block(torch.nn.Module):
    def __init__(self, x_channels, filters, strides=1, kernelSize=(3, 3)):
        super(basic_block, self).__init__()

        self.direct_layers = torch.nn.Sequential(
            torch.nn.Conv2d(x_channels, filters, kernel_size=kernelSize,
                            stride=strides, padding=1),  # paper say down sampling,
            # Batch Normalization
            torch.nn.BatchNorm2d(filters, eps=1e-05,
                                 momentum=0.1, affine=True),
            torch.nn.ReLU(),
            torch.nn.Conv2d(filters, filters,
                            kernel_size=kernelSize, stride=1, padding='same'),
            # Batch Normalization
            torch.nn.BatchNorm2d(filters, eps=1e-05,
                                 momentum=0.1, affine=True),
        )

        self.identity_layer = torch.nn.Sequential()

        if strides != 1:
            self.identity_layer = torch.nn.Conv2d(
                x_channels, filters, kernel_size=1, stride=strides)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        identity = self.identity_layer(x)
        x = self.direct_layers(x)

        x += identity

        x = self.relu(x)
        return x


class ResNet18(torch.nn.Module):
    def __init__(self, input_shape, filters):
        super(ResNet18, self).__init__()
        C, H, W = input_shape

        # self.conv0_x = torch.nn.Sequential(
        #     torch.nn.Conv2d(C, 8, (3,3), stride=1),
        #     torch.nn.MaxPool2d((3,3), stride= 1),
        #     torch.nn.Conv2d(8, 16, (3,3), stride=1),
        #     torch.nn.MaxPool2d((3,3), stride= 1),
        #     torch.nn.Conv2d(16, 64, (3,3), stride=1),
        #     torch.nn.MaxPool2d((3,3), stride= 1),
        # )
        self.conv0_x = torch.nn.Sequential(
            torch.nn.Conv2d(C, filters, (7, 7), stride=2),
            torch.nn.MaxPool2d((3, 3), stride=2),
        )

        # self.conv1 = torch.nn.Conv2d(C, 64, (7,7), stride=2)
        # self.maxpooling = torch.nn.MaxPool2d((3,3), stride=2)
        # input (3, 450, 450)
        self.conv2_x = torch.nn.Sequential(
            # stride=1 non-down sampling
            basic_block(filters, filters, 1, (3, 3)),
            basic_block(filters, filters, 1, (3, 3)),
        )
        self.conv3_x = torch.nn.Sequential(
            # stride=2 down sampling
            basic_block(filters, filters * 2, 2, (3, 3)),
            basic_block(filters * 2, filters * 2, 1, (3, 3)),
        )
        self.conv4_x = torch.nn.Sequential(
            # stride=2 down sampling
            basic_block(filters * 2, filters * 4, 2, (3, 3)),
            basic_block(filters * 4, filters * 4, 1, (3, 3)),
        )
        self.conv5_x = torch.nn.Sequential(
            # stride=2 down sampling
            basic_block(filters * 4, filters * 8, 2, (3, 3)),
            basic_block(filters * 8, filters * 8, 1, (3, 3)),
        )

        self.GlobalAvgPooling = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.Flatten = torch.nn.Flatten(1)
        # self.dropout = torch.nn.Dropout(p = 0.5)
        self.Dense = torch.nn.Linear(
            in_features=filters * 8*1*1, out_features=1, bias=True)
        self.Sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.conv0_x(x)
        # x = self.conv1(x)
        # x = self.maxpooling(x)
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        x = self.GlobalAvgPooling(x)
        x = self.Flatten(x)
        # x = self.dropout(x)
        x = self.Dense(x)
        x = self.Sigmoid(x)
        # print(x.shape)
        return x


class ResNet152(torch.nn.Module):
    def __init__(self, input_shape, filters):
        super(ResNet152, self).__init__()
        self.C, _, _ = input_shape

        # self.conv0_x = torch.nn.Sequential(
        #     torch.nn.Conv2d(C, 8, (3,3), stride=1),
        #     torch.nn.MaxPool2d((3,3), stride= 1),
        #     torch.nn.Conv2d(8, 16, (3,3), stride=1),
        #     torch.nn.MaxPool2d((3,3), stride= 1),
        #     torch.nn.Conv2d(16, 64, (3,3), stride=1),
        #     torch.nn.MaxPool2d((3,3), stride= 1),
        # )
        self.conv0_x = torch.nn.Sequential(
            torch.nn.Conv2d(self.C, filters, (7, 7), stride=2),
            torch.nn.BatchNorm2d(filters, eps=1e-05,
                                 momentum=0.1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d((3, 3), stride=2),
        )
        # self.conv1 = torch.nn.Conv2d(C, 64, (7,7), stride=2)
        # self.maxpooling = torch.nn.MaxPool2d((3,3), stride=2)
        # input (3, 450, 450)

        self.C = filters

        self.conv2_x = self._make_layer(bottleneck_block, filters, 3, stride=1)

        self.conv3_x = self._make_layer(
            bottleneck_block, filters*2, 8, stride=2)

        self.conv4_x = self._make_layer(
            bottleneck_block, filters*4, 36, stride=2)

        self.conv5_x = self._make_layer(
            bottleneck_block, filters*8, 3, stride=2)

        self.GlobalAvgPooling = torch.nn.AdaptiveAvgPool2d((1, 1))
        # self.GlobalAvgPooling = torch.nn.AvgPool2d(7, stride=1)
        self.Flatten = torch.nn.Flatten(1)
        self.dropout = torch.nn.Dropout(p=0.5, inplace=True)
        # self.Dense = torch.nn.Linear(in_features=self.C, out_features=1, bias=True)
        self.Dense = torch.nn.Linear(
            in_features=self.C, out_features=1, bias=True)
        self.Sigmoid = torch.nn.Sigmoid()

    def _make_layer(self, block, first_filters, num_blocks, stride):
        '''
        torch.nn.Sequential(
            # x1
            bottleneck_block(filters, first_filters=filters, strides=1), # stride=1 non-down sampling
            # x2
            bottleneck_block(filters*4, first_filters=filters, strides=1), # stride=1 
            # x3
            bottleneck_block(filters*4, first_filters=filters, strides=1), # stride=1 
        )
        '''
        # block(input_channel, first_filters, stride)
        layers = []
        layers.append(
            block(self.C, first_filters=first_filters, strides=stride))

        self.C = first_filters * 4

        for _ in range(num_blocks-1):
            layers.append(
                block(self.C, first_filters=first_filters, strides=1))

        return torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv0_x(x)
        # x = self.conv1(x)
        # x = self.maxpooling(x)
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        x = self.GlobalAvgPooling(x)
        x = self.Flatten(x)
        x = self.dropout(x)
        x = self.Dense(x)
        x = self.Sigmoid(x)
        # print(x.shape)
        return x


class ResNet50(torch.nn.Module):
    def __init__(self, input_shape, filters):
        super(ResNet50, self).__init__()
        self.C, _, _ = input_shape

        # self.conv0_x = torch.nn.Sequential(
        #     torch.nn.Conv2d(C, 8, (3,3), stride=1),
        #     torch.nn.MaxPool2d((3,3), stride= 1),
        #     torch.nn.Conv2d(8, 16, (3,3), stride=1),
        #     torch.nn.MaxPool2d((3,3), stride= 1),
        #     torch.nn.Conv2d(16, 64, (3,3), stride=1),
        #     torch.nn.MaxPool2d((3,3), stride= 1),
        # )
        self.conv0_x = torch.nn.Sequential(
            torch.nn.Conv2d(self.C, filters, (7, 7), stride=2),
            torch.nn.BatchNorm2d(filters, eps=1e-05,
                                 momentum=0.1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d((3, 3), stride=2),
        )
        # self.conv1 = torch.nn.Conv2d(C, 64, (7,7), stride=2)
        # self.maxpooling = torch.nn.MaxPool2d((3,3), stride=2)
        # input (3, 450, 450)

        self.C = filters

        self.conv2_x = self._make_layer(bottleneck_block, filters, 3, stride=1)

        self.conv3_x = self._make_layer(
            bottleneck_block, filters*2, 4, stride=2)

        self.conv4_x = self._make_layer(
            bottleneck_block, filters*4, 6, stride=2)

        self.conv5_x = self._make_layer(
            bottleneck_block, filters*8, 3, stride=2)

        self.GlobalAvgPooling = torch.nn.AdaptiveAvgPool2d((1, 1))
        # self.GlobalAvgPooling = torch.nn.AvgPool2d(7, stride=1)
        self.Flatten = torch.nn.Flatten(1)
        self.dropout = torch.nn.Dropout(p=0.5, inplace=True)
        # self.Dense = torch.nn.Linear(in_features=self.C, out_features=1, bias=True)
        self.Dense = torch.nn.Linear(
            in_features=self.C, out_features=1, bias=True)
        self.Sigmoid = torch.nn.Sigmoid()

    def _make_layer(self, block, first_filters, num_blocks, stride):
        '''
        torch.nn.Sequential(
            # x1
            bottleneck_block(filters, first_filters=filters, strides=1), # stride=1 non-down sampling
            # x2
            bottleneck_block(filters*4, first_filters=filters, strides=1), # stride=1 
            # x3
            bottleneck_block(filters*4, first_filters=filters, strides=1), # stride=1 
        )
        '''
        # block(input_channel, first_filters, stride)
        layers = []
        layers.append(
            block(self.C, first_filters=first_filters, strides=stride))

        self.C = first_filters * 4

        for _ in range(num_blocks-1):
            layers.append(
                block(self.C, first_filters=first_filters, strides=1))

        return torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv0_x(x)
        # x = self.conv1(x)
        # x = self.maxpooling(x)
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        x = self.GlobalAvgPooling(x)
        x = self.Flatten(x)
        x = self.dropout(x)
        x = self.Dense(x)
        x = self.Sigmoid(x)
        # print(x.shape)
        return x
