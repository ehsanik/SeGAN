
require 'torch'
require 'nn'
require 'optim'
util = paths.dofile('../util/util.lua')
require 'image'
require 'models/models'
require 'gnuplot'
require 'cunn'
require 'cudnn'
require 'xlua'
require 'math'
require 'gnuplot'
require 'sys'
require 'image'
debugger = require 'fb.debugger'
require('ROI.ROI')
require('ROI.ROIPooling')
require('networks/SeGANCri')


local input_nc = 3
local output_nc = 3 
cmd.no_masks_texture = true
-- translation direction
local idx_A = nil
local idx_B = nil
idx_A = {1, input_nc}
idx_B = {input_nc+1, input_nc+output_nc}
local ndf = opt.ndf
local ngf = opt.ngf
local real_label = 1
local fake_label = 0
local netG
local netD
local criterion 
local criterionAE
local optimStateG
local optimStateD
local real_A = torch.Tensor(opt.batchSize, input_nc, opt.fineSize, opt.fineSize)
local real_B = torch.Tensor(opt.batchSize, output_nc, opt.fineSize, opt.fineSize)
local fake_B = torch.Tensor(opt.batchSize, output_nc, opt.fineSize, opt.fineSize)
local real_AB = torch.Tensor(opt.batchSize, output_nc + input_nc*opt.condition_GAN, opt.fineSize, opt.fineSize)
local fake_AB = torch.Tensor(opt.batchSize, output_nc + input_nc*opt.condition_GAN, opt.fineSize, opt.fineSize)
local errD, errG, errL1 = 0, 0, 0
local epoch_tm = torch.Timer()
local tm = torch.Timer()
local data_tm  = torch.Timer()
local parametersD, gradParametersD
local parametersG, gradParametersG
local images, resized_full, resized_amodal, resized_modal
local df_do_AE, df_do, df_dg, net


-- Applying Equation (1) from paper to the output of segmentation network
function get_batch_texturing()
  
  if cmd.no_masks_texture then
      local weights = get_mask(resized_modal, resized_amodal)
      for ch=1,3 do
          if cmd.old_no_masks then
              config.red = config.blue
          end
          (images[{{},{ch},{},{}}])[torch.eq(weights, 2)] = config.red[ch];
          (images[{{},{ch},{},{}}])[torch.eq(weights, 0)] = config.blue[ch]
      end
  end
  return images, resized_full, resized_amodal, resized_modal
end



-- function adapted from https://github.com/phillipi/pix2pix
local function defineG(input_nc, output_nc, ngf)
    local netG = nil
    if     opt.which_model_netG == "encoder_decoder" then netG = defineG_encoder_decoder(input_nc, output_nc, ngf)
    elseif opt.which_model_netG == "unet" then netG = defineG_unet(input_nc, output_nc, ngf)
    elseif opt.which_model_netG == "unet_128" then netG = defineG_unet_128(input_nc, output_nc, ngf)
    else error("unsupported netG model")
    end

      netG:apply(weights_init)
  
    return netG
end

-- function adapted from https://github.com/phillipi/pix2pix
local function defineD(input_nc, output_nc, ndf)
    local netD = nil
    if opt.condition_GAN==1 then
        input_nc_tmp = input_nc
    else
        input_nc_tmp = 0 -- only penalizes structure in output channels
    end
    
    if     opt.which_model_netD == "basic" then netD = defineD_basic(input_nc_tmp, output_nc, ndf)
    elseif opt.which_model_netD == "n_layers" then netD = defineD_n_layers(input_nc_tmp, output_nc, ndf, opt.n_layers_D)
    else error("unsupported netD model")
    end
    
      netD:apply(weights_init)
    
    
    return netD
end

local function init_test()
  print('load texture network for test')
  opt.net = cmd.continue_texture .. '_net_G.t7'
    net = util.load(opt.net, opt)
    net:evaluate()
    require 'cunn'
    require 'cudnn'
    net = util.cudnn(net)
    net:cuda()

end

-- Initializing the network and load the saved weights in case of testing or finetuning
local function init_networks()

  -- load saved models and finetune
  if opt.continue_train == 1  then
    local continue_name = cmd.continue_texture
      print('loading previously trained netG...')
     netG = util.load(continue_name .. '_net_G.t7', opt)
     print('loading previously trained netD...')
     netD = util.load(continue_name .. '_net_D.t7', opt)
  else
    print('define model netG...')
    netG = defineG(input_nc, output_nc, ngf)
    print('define model netD...')
    netD = defineD(input_nc, output_nc, ndf)
  end

  print(netG)
  print(netD)

  criterion = nn.BCECriterion()
  criterionAE = nn.AbsCriterion()
  ---------------------------------------------------------------------------
  optimStateG = {
     learningRate = opt.lr,
     beta1 = opt.beta1,
  }
  optimStateD = {
     learningRate = opt.lr,
     beta1 = opt.beta1,
  }

  if opt.gpu > 0 then
     print('transferring to gpu...')
     require 'cunn'
     cutorch.setDevice(opt.gpu)
     real_A = real_A:cuda();
     real_B = real_B:cuda(); fake_B = fake_B:cuda();
     real_AB = real_AB:cuda(); fake_AB = fake_AB:cuda();
     if opt.cudnn==1 then
        netG = util.cudnn(netG); netD = util.cudnn(netD);
     end
     netD:cuda(); netG:cuda(); criterion:cuda(); criterionAE:cuda();
     print('done')
  else
      print('running model on CPU')
  end

  parametersD, gradParametersD = netD:getParameters()
  parametersG, gradParametersG = netG:getParameters()

end

-- Load the data and create fake output by forwarding the data to generator network to train discriminator
-- function adapted from https://github.com/phillipi/pix2pix
local function createRealFake()
    -- load real
    data_tm:reset(); data_tm:resume()
    local input, output, resized_amodal, resized_modal = get_batch_texturing()

    weight_mask = get_mask(resized_modal, resized_amodal)
    data_tm:stop()
    
    real_A:copy(input)
    real_B:copy(output)
    
    if opt.condition_GAN==1 then
        real_AB = torch.cat(real_A,real_B,2)
    else
        real_AB = real_B -- unconditional GAN, only penalizes structure in B
    end
      
    -- create fake
    fake_B = netG:forward(real_A)
    
    if opt.condition_GAN==1 then
        fake_AB = torch.cat(real_A,fake_B,2)
    else
        fake_AB = fake_B -- unconditional GAN, only penalizes structure in B
    end
end

-- create closure to evaluate f(X) and df/dX of discriminator
-- function adapted from https://github.com/phillipi/pix2pix
local fDx = function(x)
    netD:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
    netG:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
    
    gradParametersD:zero()
    
    -- Real
    local output = netD:forward(real_AB)
    if not label then
      label = torch.FloatTensor(output:size()):fill(real_label)
      if opt.gpu>0 then 
        label = label:cuda()
      end
    end
    label:fill(real_label)
    
    local errD_real = criterion:forward(output, label)
    local df_do = criterion:backward(output, label)
    netD:backward(real_AB, df_do)
    
    -- Fake
    local output = netD:forward(fake_AB)
    label:fill(fake_label)
    local errD_fake = criterion:forward(output, label)
    local df_do = criterion:backward(output, label)
    netD:backward(fake_AB, df_do)

    errD = (errD_real + errD_fake)/2
    
    return errD, gradParametersD
end

-- create closure to evaluate f(X) and df/dX of generator
-- function adapted from https://github.com/phillipi/pix2pix
local fGx = function(x)
    netD:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
    netG:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
    
    gradParametersG:zero()
    
    -- GAN loss
    if not df_dg then 
      df_dg = torch.zeros(fake_B:size())
      if opt.gpu>0 then 
        df_dg = df_dg:cuda();
      end
    end
    df_dg:fill(0)
    
    if opt.use_GAN==1 then
       local output = netD.output -- netD:forward{input_A,input_B} was already executed in fDx, so save computation
       if not label then
        label = torch.FloatTensor(output:size()):fill(real_label) -- fake labels are real for generator cost
         if opt.gpu>0 then 
            label = label:cuda();
            end
        end
       label:fill(real_label)
       errG = criterion:forward(output, label)
       local df_do = criterion:backward(output, label)
       df_dg = netD:updateGradInput(fake_AB, df_do):narrow(2,fake_AB:size(2)-output_nc+1, output_nc)
    else
        errG = 0
    end
    
    -- unary loss
    if not df_do_AE then
      df_do_AE = torch.zeros(fake_B:size())
      if opt.gpu>0 then 
        df_do_AE = df_do_AE:cuda();
      end
    end
    df_do_AE:fill(0)
    if opt.use_L1==1 then 
      if opt.myloss then
        local criterionSeGAN = nn.SeGANCriterion(output_nc, weight_mask):cuda()
        errL1 = criterionSeGAN:forward(fake_B, real_B)
        df_do_AE = criterionSeGAN:backward(fake_B, real_B)
      else
        errL1 = criterionAE:forward(fake_B, real_B)
        df_do_AE = criterionAE:backward(fake_B, real_B)
      end
    else
        errL1 = 0
    end
    
    netG:backward(real_A, df_dg + df_do_AE:mul(opt.lambda))
    
    return errG, gradParametersG
end


-- Save the model for texture generation network
function texture_model_save()
  print('saving')
  torch.save(opt.saveadr .. opt.name_texture .. '_net_G.t7', netG:clearState())
  torch.save(opt.saveadr .. opt.name_texture .. '_net_D.t7', netD:clearState())
end

-- Forward for texture network, return the gradients as well in case of end to end training
function train_texture(im, full, amodal, modal)
  images = im:clone()
  resized_full = full
  resized_amodal = amodal
  resized_modal = modal

  local epoch = epoch_texture
  createRealFake()
  if opt.use_GAN==1 then optim.adam(fDx, parametersD, optimStateD) end
  optim.adam(fGx, parametersG, optimStateG)
  print(('For texturing  Err_G: %.4f  Err_D: %.4f  ErrL1: %.4f'):format(errG, errD, errL1))     
  collectgarbage('collect')
  local gradient=netG:get(1).gradInput
  return fake_B, gradient[{{},{1},{},{}}] 
end

function make_param_grads_texture_nill()

  parametersD, gradParametersD = nil, nil -- nil them to avoid spiking memory
  parametersG, gradParametersG = nil, nil

  parametersD, gradParametersD = netD:getParameters() -- reflatten the params and get them
  parametersG, gradParametersG = netG:getParameters()
end

-- Just forward pass (no backward) for the texture network
function test_texture(im, full, amodal, modal)
  images = im:clone()
  resized_full = full
  resized_amodal = amodal
  resized_modal = modal
 
  local input, output, resized_amodal, resized_modal = get_batch_texturing()
  local real_ctx = input:cuda()
  local pred_center
  pred_center = net:forward(real_ctx)

   return pred_center

end

if cmd.istrain == 'train' and cmd.end2end then init_networks() end
if cmd.end2end and cmd.istrain ~= 'train' then init_test() end

