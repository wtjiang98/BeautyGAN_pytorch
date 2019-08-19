import torch
import torch.nn.init as init
from torch.autograd import Variable
from torchvision.utils import save_image

import itertools
import os
import time
import datetime

import tools.plot as plot_fig
import net
from ops.loss_added import GANLoss

class Solver_cycleGAN(object):
    """
        solver to reproduce the cycleGAN
    """
    def __init__(self, data_loaders, config, dataset_config):
        # dataloader
        self.checkpoint = config.checkpoint
        # Hyper-parameteres
        self.g_lr = config.G_LR
        self.d_lr = config.D_LR
        self.ndis = config.ndis
        self.num_epochs = config.num_epochs  # set 200
        self.num_epochs_decay = config.num_epochs_decay

        # Training settings
        self.snapshot_step = config.snapshot_step
        self.log_step = config.log_step
        self.vis_step = config.vis_step

        #training setting
        self.task_name = config.task_name

        # Data loader
        self.data_loader_train = data_loaders[0]
        self.data_loader_test = data_loaders[1]

        # Model hyper-parameters
        self.img_size = config.img_size
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.g_repeat_num = config.g_repeat_num
        self.d_repeat_num = config.d_repeat_num

        # Hyper-parameteres
        self.lambda_idt = config.lambda_idt
        self.lambda_A = config.lambda_A
        self.lambda_B = config.lambda_B

        self.beta1 = config.beta1
        self.beta2 = config.beta2

        # Test settings
        self.test_model = config.test_model

        # Path
        self.log_path = config.log_path + '_' + config.task_name
        self.vis_path = config.vis_path + '_' + config.task_name
        self.snapshot_path = config.snapshot_path + '_' + config.task_name
        self.result_path = config.vis_path + '_' + config.task_name

        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        if not os.path.exists(self.vis_path):
            os.makedirs(self.vis_path)
        if not os.path.exists(self.snapshot_path):
            os.makedirs(self.snapshot_path)


        self.build_model()
        # Start with trained model
        if self.checkpoint:
            self.load_checkpoint()

        #for recording
        self.start_time = time.time()
        self.e = 0
        self.i = 0
        self.loss = {}

        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        if not os.path.exists(self.vis_path):
            os.makedirs(self.vis_path)
        if not os.path.exists(self.snapshot_path):
            os.makedirs(self.snapshot_path)

    def print_network(self, model, name):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(name)
        print(model)
        print("The number of parameters: {}".format(num_params))

    def update_lr(self, g_lr, d_lr):
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_A_optimizer.param_groups:
            param_group['lr'] = d_lr
        for param_group in self.d_B_optimizer.param_groups:
            param_group['lr'] = d_lr

    def log_terminal(self):
        elapsed = time.time() - self.start_time
        elapsed = str(datetime.timedelta(seconds=elapsed))

        log = "Elapsed [{}], Epoch [{}/{}], Iter [{}/{}]".format(
            elapsed, self.e+1, self.num_epochs, self.i+1, self.iters_per_epoch)

        for tag, value in self.loss.items():
            log += ", {}: {:.4f}".format(tag, value)
        print(log)

    def save_models(self):
        torch.save(self.G_A.state_dict(),
                   os.path.join(self.snapshot_path, '{}_{}_G_A.pth'.format(self.e + 1, self.i + 1)))
        torch.save(self.G_B.state_dict(),
                   os.path.join(self.snapshot_path, '{}_{}_G_B.pth'.format(self.e + 1, self.i + 1)))
        torch.save(self.D_A.state_dict(),
                   os.path.join(self.snapshot_path, '{}_{}_D_A.pth'.format(self.e + 1, self.i + 1)))
        torch.save(self.D_B.state_dict(),
                   os.path.join(self.snapshot_path, '{}_{}_D_B.pth'.format(self.e + 1, self.i + 1)))

    def weights_init_xavier(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            init.xavier_normal(m.weight.data, gain=1.0)
        elif classname.find('Linear') != -1:
            init.xavier_normal(m.weight.data, gain=1.0)

    def to_var(self, x, requires_grad=True):
        if torch.cuda.is_available():
            x = x.cuda()
        if not requires_grad:
            return Variable(x, requires_grad=requires_grad)
        else:
            return Variable(x)

    def denorm(self, x):
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def load_checkpoint(self):
        self.G_A.load_state_dict(torch.load(os.path.join(
            self.snapshot_path, '{}_G_A.pth'.format(self.checkpoint))))
        self.G_B.load_state_dict(torch.load(os.path.join(
            self.snapshot_path, '{}_G_B.pth'.format(self.checkpoint))))
        self.D_A.load_state_dict(torch.load(os.path.join(
            self.snapshot_path, '{}_D_A.pth'.format(self.checkpoint))))
        self.D_B.load_state_dict(torch.load(os.path.join(
            self.snapshot_path, '{}_D_B.pth'.format(self.checkpoint))))
        print('loaded trained models (step: {})..!'.format(self.checkpoint))

    def build_model(self):
        # Define generators and discriminators
        self.G_A = net.Generator(self.g_conv_dim, self.g_repeat_num) 
        self.G_B = net.Generator(self.g_conv_dim, self.g_repeat_num)
        self.D_A = net.Discriminator(self.img_size, self.d_conv_dim, self.d_repeat_num)
        self.D_B = net.Discriminator(self.img_size, self.d_conv_dim, self.d_repeat_num)
        self.criterionL1 = torch.nn.L1Loss()
        self.criterionGAN = GANLoss(use_lsgan=True, tensor =torch.cuda.FloatTensor)

        # Optimizers
        self.g_optimizer = torch.optim.Adam(itertools.chain(self.G_A.parameters(), self.G_B.parameters()),
                                                self.g_lr, [self.beta1, self.beta2])
        self.d_A_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.D_A.parameters()), self.d_lr, [self.beta1, self.beta2])
        self.d_B_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.D_B.parameters()), self.d_lr, [self.beta1, self.beta2])

        self.G_A.apply(self.weights_init_xavier)
        self.D_A.apply(self.weights_init_xavier)
        self.G_B.apply(self.weights_init_xavier)
        self.D_B.apply(self.weights_init_xavier)

        # Print networks
      #  self.print_network(self.E, 'E')
        self.print_network(self.G_A, 'G_A')
        self.print_network(self.D_A, 'D_A')
        self.print_network(self.G_B, 'G_B')
        self.print_network(self.D_B, 'D_B')

        if torch.cuda.is_available():
            self.G_A.cuda()
            self.G_B.cuda()
            self.D_A.cuda()
            self.D_B.cuda()

    def train(self):
        """Train StarGAN within a single dataset."""
        # The number of iterations per epoch
        self.iters_per_epoch = len(self.data_loader_train)
        # Start with trained model if exists
        g_lr = self.g_lr
        d_lr = self.d_lr
        if self.checkpoint:
            start = int(self.checkpoint.split('_')[0])
        else:
            start = 0
        # Start training
        self.start_time = time.time()
        for self.e in range(start, self.num_epochs):
            for self.i, (img_A, img_B, _, _) in enumerate(self.data_loader_train):
                # Convert tensor to variable
                org_A = self.to_var(img_A, requires_grad=False)
                ref_B = self.to_var(img_B, requires_grad=False)

                # ================== Train D ================== #
                # training D_A
                # Real
                out = self.D_A(ref_B)
                d_loss_real = self.criterionGAN(out, True)
                # Fake
                fake = self.G_A(org_A)
                fake = Variable(fake.data)
                fake = fake.detach()
                out = self.D_A(fake)
                #d_loss_fake = self.get_D_loss(out, "fake")
                d_loss_fake =  self.criterionGAN(out, False)
               
                # Backward + Optimize
                d_loss = (d_loss_real + d_loss_fake) * 0.5
                self.d_A_optimizer.zero_grad()
                d_loss.backward(retain_graph=True)
                self.d_A_optimizer.step()

                # Logging
                self.loss = {}
                self.loss['D-A-loss_real'] = d_loss_real.item()

                # training D_B
                # Real
                out = self.D_B(org_A)
                d_loss_real = self.criterionGAN(out, True)
                # Fake
                fake = self.G_B(ref_B)
                fake = Variable(fake.data)
                fake = fake.detach()
                out = self.D_B(fake)
                #d_loss_fake = self.get_D_loss(out, "fake")
                d_loss_fake =  self.criterionGAN(out, False)
               
                # Backward + Optimize
                d_loss = (d_loss_real + d_loss_fake) * 0.5
                self.d_B_optimizer.zero_grad()
                d_loss.backward(retain_graph=True)
                self.d_B_optimizer.step()

                # Logging
                self.loss['D-B-loss_real'] = d_loss_real.item()

                # ================== Train G ================== #
                if (self.i + 1) % self.ndis == 0:
                    # adversarial loss, i.e. L_trans,v in the paper 

                    # identity loss
                    if self.lambda_idt > 0:
                        # G_A should be identity if ref_B is fed
                        idt_A = self.G_A(ref_B)
                        loss_idt_A = self.criterionL1(idt_A, ref_B) * self.lambda_B * self.lambda_idt
                        # G_B should be identity if org_A is fed
                        idt_B = self.G_B(org_A)
                        loss_idt_B = self.criterionL1(idt_B, org_A) * self.lambda_A * self.lambda_idt
                        g_loss_idt = loss_idt_A + loss_idt_B
                    else:
                        g_loss_idt = 0
                        
                    # GAN loss D_A(G_A(A))
                    fake_B = self.G_A(org_A)
                    pred_fake = self.D_A(fake_B)
                    g_A_loss_adv =  self.criterionGAN(pred_fake, True)
                    #g_loss_adv = self.get_G_loss(out)

                    # GAN loss D_B(G_B(B))
                    fake_A = self.G_B(ref_B)
                    pred_fake = self.D_B(fake_A)
                    g_B_loss_adv = self.criterionGAN(pred_fake, True)

                    # Forward cycle loss
                    rec_A = self.G_B(fake_B)
                    g_loss_rec_A = self.criterionL1(rec_A, org_A) * self.lambda_A

                    # Backward cycle loss
                    rec_B = self.G_A(fake_A)
                    g_loss_rec_B = self.criterionL1(rec_B, ref_B) * self.lambda_B

                    # Combined loss
                    g_loss = g_A_loss_adv + g_B_loss_adv + g_loss_rec_A + g_loss_rec_B + g_loss_idt
                    
                    self.g_optimizer.zero_grad()
                    g_loss.backward(retain_graph=True)
                    self.g_optimizer.step()

                    # Logging
                    self.loss['G-A-loss_adv'] = g_A_loss_adv.item()
                    self.loss['G-B-loss_adv'] = g_A_loss_adv.item()
                    self.loss['G-loss_org'] = g_loss_rec_A.item()
                    self.loss['G-loss_ref'] = g_loss_rec_B.item()
                    self.loss['G-loss_idt'] = g_loss_idt.item()

                # Print out log info
                if (self.i + 1) % self.log_step == 0:
                    self.log_terminal()

                #plot the figures
                for key_now in self.loss.keys():
                    plot_fig.plot(key_now, self.loss[key_now])

                #save the images
                if (self.i + 1) % self.vis_step == 0:
                    print("Saving middle output...")
                    self.vis_train([org_A, ref_B, fake_A, fake_B, rec_A, rec_B])
                    self.vis_test()

                # Save model checkpoints
                if (self.i + 1) % self.snapshot_step == 0:
                    self.save_models()

                if (self.i % 100 == 99):
                    plot_fig.flush(self.task_name)

                plot_fig.tick()
            
            # Decay learning rate
            if (self.e+1) > (self.num_epochs - self.num_epochs_decay):
                g_lr -= (self.g_lr / float(self.num_epochs_decay))
                d_lr -= (self.d_lr / float(self.num_epochs_decay))
                self.update_lr(g_lr, d_lr)
                print('Decay learning rate to g_lr: {}, d_lr:{}.'.format(g_lr, d_lr))

    def vis_train(self, img_train_list):
        # saving training results
        mode = "train_vis"
        img_train_list = torch.cat(img_train_list, dim=3)
        result_path_train = os.path.join(self.result_path, mode)
        if not os.path.exists(result_path_train):
            os.mkdir(result_path_train)
        save_path = os.path.join(result_path_train, '{}_{}_fake.jpg'.format(self.e, self.i))
        save_image(self.denorm(img_train_list.data), save_path, normalize=True)

    def vis_test(self):
        # saving test results
        mode = "test_vis"
        for i, (img_A, img_B) in enumerate(self.data_loader_test):
            real_org = self.to_var(img_A)
            real_ref = self.to_var(img_B)

            image_list = []
            image_list.append(real_org)
            image_list.append(real_ref)

            # Get makeup result
            fake_A = self.G_A(real_org)
            fake_B = self.G_B(real_ref)
            rec_A = self.G_B(fake_A)
            rec_B = self.G_A(fake_B)

            image_list.append(fake_A)
            image_list.append(fake_B)
            image_list.append(rec_A)
            image_list.append(rec_B)

            image_list = torch.cat(image_list, dim=3)
            vis_train_path = os.path.join(self.result_path, mode)
            result_path_now = os.path.join(vis_train_path, "epoch" + str(self.e))
            if not os.path.exists(result_path_now):
                os.makedirs(result_path_now)
            save_path = os.path.join(result_path_now, '{}_{}_{}_fake.jpg'.format(self.e, self.i, i + 1))
            save_image(self.denorm(image_list.data), save_path, normalize=True)
            #print('Translated test images and saved into "{}"..!'.format(save_path))

    def test(self):
        # Load trained parameters
        G_A_path = os.path.join(self.snapshot_path, '{}_G_A.pth'.format(self.test_model))
        G_B_path = os.path.join(self.snapshot_path, '{}_G_B.pth'.format(self.test_model))
        self.G_A.load_state_dict(torch.load(G_A_path))
        self.G_A.eval()
        self.G_B.load_state_dict(torch.load(G_B_path))
        self.G_B.eval()
        for i, (img_A, img_B) in enumerate(self.data_loader_test):
            real_org = self.to_var(img_A)
            real_ref = self.to_var(img_B)

            image_list = []
            image_list.append(real_org)
            image_list.append(real_ref)

            # Get makeup result
            fake_A = self.G_A(real_org)
            fake_B = self.G_B(real_ref)
            rec_A = self.G_B(fake_A)
            rec_B = self.G_A(fake_B)

            image_list.append(fake_A)
            image_list.append(fake_B)
            image_list.append(rec_A)
            image_list.append(rec_B)

            image_list = torch.cat(image_list, dim=3)
            save_path = os.path.join(self.result_path, '{}_{}_{}_fake.png'.format(self.e, self.i, i + 1))
            save_image(self.denorm(image_list.data), save_path, nrow=1, padding=0, normalize=True)
            print('Translated test images and saved into "{}"..!'.format(save_path))
