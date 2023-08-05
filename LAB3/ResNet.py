import torch
# print("Please define your ResNet in this file.")

class basic_block(torch.nn.Module):
    def __init__(self, x_channels, filters, strides=1, kernelSize=(3,3)):
        super(basic_block, self).__init__()
        
        self.direct_layers = torch.nn.Sequential(
            torch.nn.Conv2d(x_channels, filters, kernel_size=kernelSize, stride=strides, padding=1), # paper say down sampling,

            torch.nn.ReLU(),

            torch.nn.Conv2d(filters, filters, kernel_size=kernelSize, stride=1, padding='same')
        )

        self.identity_layer = torch.nn.Conv2d(x_channels, filters, kernel_size=1, stride=strides)

        self.relu = torch.nn.ReLU()

        self.strides = strides

    def forward(self, x):
        direct_x = self.direct_layers(x)
        
        if self.strides == 1:
            # no down sampling
            x = direct_x + x
        else:
            identity = self.identity_layer(x)
            x = direct_x + identity
        
        x = self.relu(x)
        return x


    
# def bottleneck_block(x, filters_list, strides):
#     channels = x.shape[0]
#     direct_x = torch.nn.Conv2d(channels, 64, (1,1), stride=1, padding='same')
#     direct_x = torch.nn.ReLU()
#     direct_x = torch.nn.Conv2d(channels, 64, (3,3), stride=1, padding='same')
#     direct_x = torch.nn.Conv2d(channels, 256, (1,1), stride=1, padding='same')

#     x = direct_x + x
#     x = torch.nn.ReLU()
#     return x
    

class ResNet18(torch.nn.Module):
    def __init__(self, input_shape):
        super(ResNet18, self).__init__()
        C,H,W = input_shape

        self.conv1 = torch.nn.Conv2d(C, 64, (7,7), stride=2)
        self.maxpooling = torch.nn.MaxPool2d((3,3), stride=2)
        # input (3, 450, 450)
        self.conv2_x = torch.nn.Sequential(
            basic_block(64, 64, 1, (3,3)), # stride=1 non-down sampling
            basic_block(64, 64, 1, (3,3)),
        )
        self.conv3_x = torch.nn.Sequential(
            basic_block(64, 128, 2, (3,3)), # stride=2 down sampling
            basic_block(128, 128, 1, (3,3)),
        )
        self.conv4_x = torch.nn.Sequential(
            basic_block(128, 256, 2, (3,3)), # stride=2 down sampling
            basic_block(256, 256, 1, (3,3)),
        )
        self.conv5_x = torch.nn.Sequential(
            basic_block(256, 512, 2, (3,3)), # stride=2 down sampling
            basic_block(512, 512, 1, (3,3)),
        )

        self.GlobalAvgPooling = torch.nn.AdaptiveAvgPool2d((1,1))
        self.Flatten = torch.nn.Flatten(1)
        self.Dense = torch.nn.Linear(in_features=512*1*1, out_features=1, bias=True)
        self.Sigmoid = torch.nn.Sigmoid()
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpooling(x)
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        x = self.GlobalAvgPooling(x)
        x = self.Flatten(x)
        x = self.Dense(x)
        x = self.Sigmoid(x)
        # print(x.shape)
        return x
    
    
        