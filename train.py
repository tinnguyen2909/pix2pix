"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import time
import torch
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from models.networks import MultiheadAttentionBlock
from util.visualizer import Visualizer

if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    # Handle loading checkpoints trained without attention
    if opt.continue_train and opt.checkpoint_no_attention and opt.use_attention:
        print('Loading checkpoint trained without attention, then enabling attention...')
        
        # Temporarily disable attention to properly load the checkpoint
        opt_no_attn = TrainOptions().parse()  # Create a fresh copy of the options
        # Copy all attributes from opt to opt_no_attn
        for key, value in vars(opt).items():
            setattr(opt_no_attn, key, value)
        opt_no_attn.use_attention = False
        
        # Create model without attention and load checkpoint
        model = create_model(opt_no_attn)
        model.setup(opt_no_attn)
        print(f"Created model with use_attention = {opt_no_attn.use_attention}")
        
        # Now insert attention directly into the innermost layer
        def find_innermost_block(model):
            """Find the innermost UnetSkipConnectionBlock in a nested structure"""
            def _find_recursively(module):
                # Check if this module is the innermost UnetSkipConnectionBlock
                if hasattr(module, 'innermost') and module.innermost:
                    return module
                
                # Check all children modules
                if hasattr(module, 'model'):
                    # If model is a Sequential, check each child
                    if isinstance(module.model, torch.nn.Sequential):
                        for child in module.model:
                            result = _find_recursively(child)
                            if result is not None:
                                return result
                    # If model is another module, check it recursively
                    else:
                        return _find_recursively(module.model)
                return None
            
            return _find_recursively(model)
        
        # Get the generator network
        netG = model.netG
        if isinstance(netG, torch.nn.DataParallel):
            netG = netG.module
        
        # Find the innermost block
        innermost = find_innermost_block(netG)
        
        if innermost is not None and hasattr(innermost, 'model') and isinstance(innermost.model, torch.nn.Sequential):
            print(f"Found innermost block, inserting attention")
            
            # Get the current innermost sequential model
            innermost_seq = innermost.model
            
            # Print the current innermost structure
            print(f"Current innermost structure: {[type(m).__name__ for m in innermost_seq]}")
            
            # Look for ReLU and ConvTranspose2d in the upsampling part
            # Typical structure: [LeakyReLU, Conv2d, ReLU, ConvTranspose2d, BatchNorm2d]
            up_relu_idx = None
            up_conv_idx = None
            
            for i, module in enumerate(innermost_seq):
                if i > 1 and isinstance(module, torch.nn.ReLU):  # Skip first modules (downsampling)
                    up_relu_idx = i
                if i > 2 and isinstance(module, torch.nn.ConvTranspose2d):  # After ReLU
                    up_conv_idx = i
                    break
            
            if up_relu_idx is not None and up_conv_idx is not None:
                print(f"Found upsampling ReLU at index {up_relu_idx} and ConvTranspose2d at index {up_conv_idx}")
                
                # Create new sequential with attention inserted between ReLU and ConvTranspose2d
                new_seq = torch.nn.Sequential()
                
                # Add all modules up to and including ReLU
                for i in range(up_relu_idx + 1):
                    new_seq.add_module(str(i), innermost_seq[i])
                
                # Create and add the attention module (using the same number of channels as ConvTranspose2d input)
                conv_transpose = innermost_seq[up_conv_idx]
                if hasattr(conv_transpose, 'in_channels'):
                    channels = conv_transpose.in_channels
                    attention_heads = opt.attention_heads if hasattr(opt, 'attention_heads') else 8
                    attention = MultiheadAttentionBlock(
                        embed_dim=channels, 
                        num_heads=attention_heads,
                        dropout=0.0
                    )
                    
                    # Move the attention module to the same device as the model
                    device = next(innermost_seq.parameters()).device
                    attention = attention.to(device)
                    
                    print(f"Adding attention with {channels} channels and {attention_heads} heads on {device}")
                    new_seq.add_module(str(up_relu_idx + 1), attention)
                
                # Add remaining modules (ConvTranspose2d and onwards)
                for i in range(up_conv_idx, len(innermost_seq)):
                    new_seq.add_module(str(i + 1), innermost_seq[i])
                
                # Replace the innermost sequential with the new one
                innermost.model = new_seq
                
                # Mark model as having attention
                netG.use_attention = True
                
                print(f"New innermost structure: {[type(m).__name__ for m in innermost.model]}")
                print("Successfully inserted attention into the innermost layer")
                
                # Ensure the entire model is on the appropriate device
                if len(opt.gpu_ids) > 0 and torch.cuda.is_available():
                    model.netG.to(model.device)
                    # If using multiple GPUs, recreate the DataParallel wrapper
                    if len(opt.gpu_ids) > 1:
                        model.netG = torch.nn.DataParallel(netG, opt.gpu_ids)
                    print(f"Moved entire model to {model.device} and reset DataParallel wrapper")
            else:
                print("Could not find proper insertion points in the innermost layer")
        else:
            print("Could not find innermost block or it doesn't have a sequential model")
        
        # Setup the modified model properly (optimizers, etc.)
        # But don't try to load checkpoint again
        opt.continue_train = False
        model.setup(opt)
        # Restore continue_train for correct saving behavior
        opt.continue_train = True
    else:
        # Standard model creation without special handling
        model = create_model(opt)      # create a model given opt.model and other options
        model.setup(opt)               # regular setup: load and print networks; create schedulers
    
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        model.update_learning_rate()    # update learning rates in the beginning of every epoch.
        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
