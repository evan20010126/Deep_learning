import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler, UNet2DModel
from matplotlib import pyplot as plt
from tqdm.auto import tqdm
import json
from torchvision import transforms
import dataloader

test_only = False
load_model_parameters = True
model_path = 'third_model'
batchsize = 16
n_epochs = 50
num_train_timesteps = 1000
noise_scheduler = DDPMScheduler(num_train_timesteps=num_train_timesteps, beta_schedule='squaredcos_cap_v2')
torch.cuda.empty_cache()

device = 'mps' if torch.backends.mps.is_available(
) else 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

transform=transforms.Compose([
                                            transforms.ToPILImage(),
                                            transforms.Resize((64,64)),
                                            transforms.RandomHorizontalFlip(),
                                            # transforms.Resize((64, 64)),  # Resize the image to your desired size
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.5], [0.5]),
                                           ])
dataset = dataloader.iclevrDataset("./iclevr/", 'train', transform)

class ClassConditionedUnet(nn.Module):
  def __init__(self, num_classes=24, class_emb_size=1):
    super().__init__()
    
    # The embedding layer will map the class label to a vector of size class_emb_size
    self.class_emb = nn.Embedding(num_classes + 1, class_emb_size)

    # Self.model is an unconditional UNet with extra input channels to accept the conditioning information (the class embedding)
    self.model = UNet2DModel(
        sample_size=(64, 64),           # the target image resolution
        in_channels=3 + 3, # Additional input channels for class cond.
        out_channels=3,           # the number of output channels
        layers_per_block=2,       # how many ResNet layers to use per UNet block
        block_out_channels=(32, 64, 64), 
        # downsample_type='resnet',
        # upsample_type='resnet',
        down_block_types=( 
            "DownBlock2D",        # a regular ResNet downsampling block
            "AttnDownBlock2D",    # a ResNet downsampling block with spatial self-attention
            "AttnDownBlock2D",
        ), 
        up_block_types=(
            "AttnUpBlock2D", 
            "AttnUpBlock2D",      # a ResNet upsampling block with spatial self-attention
            "UpBlock2D",          # a regular ResNet upsampling block
          ),
        # norm_num_groups=4
        # down_block_types=( 
        #     "DownBlock2D",        # a regular ResNet downsampling block
        #     "AttnDownBlock2D",    # a ResNet downsampling block with spatial self-attention
        #     "AttnDownBlock2D",
        # ), 
        # up_block_types=(
        #     "AttnUpBlock2D", 
        #     "AttnUpBlock2D",      # a ResNet upsampling block with spatial self-attention
        #     "UpBlock2D",          # a regular ResNet upsampling block
        #   ),
    )

  # Our forward method now takes the class labels as an additional argument
  def forward(self, x, t, class_labels):
    # Shape of x:
    bs, ch, w, h = x.shape
    
    # class conditioning in right shape to add as additional input channels
    class_labels = self.class_emb(class_labels) # Map to embedding dinemsion

    # class_cond = class_cond.view(bs, class_cond.shape[1], 1, 1).expand(bs, class_cond.shape[1], w, h)
    # x is shape (bs, 1, 28, 28) and class_cond is now (bs, 4, 28, 28)
    ###################### print(class_labels.size(), "here")
    class_labels = class_labels.view(bs, class_labels.shape[1], 1, 1).expand(bs, class_labels.shape[1], w, h)
    # Net input is now x and class cond concatenated together along dimension 1
    x = torch.cat((x, class_labels), 1)

    # Feed this to the unet alongside the timestep and return the prediction
    return self.model(x, t).sample # (bs, 1, 28, 28)
  

train_dataloader = DataLoader(
    dataset=dataset, batch_size=batchsize, shuffle=True)

print(next(iter(train_dataloader))[0].size())

model = ClassConditionedUnet().to(device)
loss_fn = nn.MSELoss()
opt = torch.optim.Adam(model.parameters(), lr=1e-3) 

losses = []

if load_model_parameters:
  print("LOAD PRE MODEL")
  model.load_state_dict(torch.load(f'./{model_path}.pt'))
  model_path = model_path + '_reload'

if not test_only:
  # The training loop
  best_loss = 10
  for epoch in range(n_epochs):
      print("Epoch: ", epoch)
      for x, label in tqdm(train_dataloader):
          
          # Get some data and prepare the corrupted version
          x = x.to(device)
          label = label.to(device)
          noise = torch.randn_like(x)
          timesteps = torch.randint(0, num_train_timesteps - 1, (x.shape[0],)).long().to(device)
          noisy_x = noise_scheduler.add_noise(x, noise, timesteps)
          # print("noisy_x", noisy_x.size())##################
          # Get the model prediction
          pred = model(noisy_x, timesteps, label) # Note that we pass in the labels y

          # Calculate the loss
          loss = loss_fn(pred, noise) # How close is the output to the noise

          # Backprop and update the params:
          opt.zero_grad()
          loss.backward()
          opt.step()

          # Store the loss for later
          losses.append(loss.item())

          if best_loss > loss.item():
            torch.save(model.state_dict(), f"{model_path}_best.pt")
            best_loss = loss.item()
            
  # View the loss curve
  
  torch.save(model.state_dict(), f"{model_path}.pt")
  plt.plot(losses)
  plt.savefig(f'{model_path}.png')
  print("Complete all training")
else:
  model.load_state_dict(torch.load(f'./{model_path}.pt'))

  x = torch.randn(1, 3, 64, 64).to(device)
  y = torch.tensor([[13, 24, 24]]).to(device)
  # Sampling loop
  for i, t in tqdm(enumerate(noise_scheduler.timesteps)):

      # Get model pred
      with torch.no_grad():
          residual = model(x, t, y)  # Again, note that we pass in our labels y
      # Update sample with step
          x = noise_scheduler.step(residual, t, x).prev_sample

  # Show the results
  # print(x.detach().cpu().size())
  # print(torch.min(x.detach().cpu()))
  # print(torch.max(x.detach().cpu())) # -1 ~ 1
  # grid = torchvision.utils.make_grid((x.detach().cpu() + 1.0) / 2.0) # -1~1 mapping 到 0~1
  # torchvision.utils.save_image(grid, f'{model_path}_test.png')
  x = x.detach().cpu()
  # x = transforms.ToPILImage()(x.squeeze(0))
  # plt.imshow(x)
  fig, ax = plt.subplots(1, 1, figsize=(12, 12))
  ax.imshow(torchvision.utils.make_grid(x.clip(-1, 1), nrow=8)[0])
  plt.savefig(f'{model_path}_test.png')
