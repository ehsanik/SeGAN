--
-- code adapted from https://github.com/phillipi/pix2pix
--

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

-- create closure to evaluate f(X) and df/dX of discriminator
fDx = function(x)
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
fGx = function(x)
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
        local criterionSeGAN = nn.SeGANCriterion(opt.nc_output, weight_mask):cuda()
        errL1 = criterionSeGAN:forward(fake_B, real_B)
        df_do_AE = criterionSeGAN:backward(fake_B, real_B)
      else
        errL1 = criterionAE:forward(fake_B, real_B)
        df_do_AE = criterionAE:backward(fake_B, real_B)
      end
    else
        errL1 = 0
    end

    if cmd.end2end then    
      netG:backward(real_A, df_dg + df_do_AE:mul(opt.lambda) + gradient_end2end:mul(opt.lambda2))
    else
      netG:backward(real_A, df_dg + df_do_AE:mul(opt.lambda) )
    end

    
    return errG, gradParametersG
end


-- Adapted from https://github.com/phillipi/pix2pix
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

-- Adapted from https://github.com/phillipi/pix2pix
local function defineD(input_nc, output_nc, ndf)
    local netD = nil
    if opt.condition_GAN==1 then
      input_nc_tmp = input_nc
    else
      input_nc_tmp = 0 -- only penalizes structure in output channels
    end
    
    if opt.which_model_netD == "basic" then 
      netD = defineD_basic(input_nc_tmp, output_nc, ndf)
    elseif opt.which_model_netD == "n_layers" then 
      netD = defineD_n_layers(input_nc_tmp, output_nc, ndf, opt.n_layers_D)
    else 
      error("unsupported netD model")
    end
    netD:apply(weights_init)
    
    return netD
end


function weights_init(m)
  local name = torch.type(m)
  if name:find('Convolution') then
    m.weight:normal(0.0, 0.02)
    m.bias:fill(0)
  elseif name:find('BatchNormalization') then
    if m.weight then m.weight:normal(1.0, 0.02) end
    if m.bias then m.bias:fill(0) end
  end
end

-- Contructing the networks
function constructorTextureNet()
  
  idx_A = {1, input_nc}
  idx_B = {input_nc+1, input_nc+output_nc}

  if opt.display == 0 then opt.display = false end

  opt.manualSeed = torch.random(1, 10000) -- fix seed
  print("Random Seed: " .. opt.manualSeed)
  torch.manualSeed(opt.manualSeed)
  torch.setdefaulttensortype('torch.FloatTensor')

  data_size = #dataset
  print('Data Size  ' .. data_size)

  -- load saved models and finetune
  if opt.continue_train == 1 and not cmd.cvpr then
     local continue_name = cmd.continue_segmentation
     print('loading previously trained netG...')
     netG = util.load(opt.saveadr .. continue_name .. '_net_G.t7', opt)
     print('loading previously trained netD...')
     netD = util.load(opt.saveadr .. continue_name .. '_net_D.t7', opt)
  elseif not cmd.cvpr then
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

function train()

  local counter = 0
  epoch_texture = 1
  for epoch = 1, opt.niter do
    epoch_texture = epoch
      epoch_tm:reset()
      for i = 1, math.min(data_size, opt.ntrain), opt.batchSize do
          tm:reset()
          
          -- load a batch and run G on that batch
          createRealFake()
          if cmd.end2end then
            amodal_pred = clean_mask_output(fake_B:clone())
            output_end2end, gradient_end2end = train_texture(input_critical[{{},{1,3},{},{}}], output_critical, amodal_pred, modal_predicted) 
          end

           
          -- (1) Update D network: maximize log(D(x,y)) + log(1 - D(x,G(x)))
          if opt.use_GAN==1 then optim.adam(fDx, parametersD, optimStateD) end
          
          -- (2) Update G network: maximize log(D(x,G(x))) + L1(y,G(x))
          optim.adam(fGx, parametersG, optimStateG)

          -- display
          counter = counter + 1
          
          -- logging and display plot
          if counter % opt.print_freq == 0 then
              local loss = {errG=errG and errG or -1, errD=errD and errD or -1, errL1=errL1 and errL1 or -1}
              local curItInBatch = ((i-1) / opt.batchSize)
              local totalItInBatch = math.floor(math.min(data_size, opt.ntrain) / opt.batchSize)
              log(('Epoch: [%d][%8d / %8d]\t Time: %.3f  DataTime: %.3f  '
                      .. '  Err_G: %.4f  Err_D: %.4f  ErrL1: %.4f'):format(
                       epoch, curItInBatch, totalItInBatch,
                       tm:time().real / opt.batchSize, data_tm:time().real / opt.batchSize,
                       errG, errD, errL1))
              collectgarbage('collect')
              
          end
          
          -- save latest model
          paths.mkdir(opt.saveadr )

          if counter % opt.save_latest_freq == 0 then
              print(('saving the latest model (epoch %d, iters %d)'):format(epoch, counter))
              torch.save(paths.concat(opt.checkpoints_dir, opt.name, 'latest_net_G.t7'), netG:clearState())
              torch.save(paths.concat(opt.checkpoints_dir, opt.name, 'latest_net_D.t7'), netD:clearState())
              flushit()
          end
          collectgarbage('collect')
          
      end
      
      if epoch % opt.save_epoch_freq == 0 then
          log('saving')
          torch.save(opt.saveadr .. opt.name .. '_net_G.t7', netG:clearState())
          torch.save(opt.saveadr .. opt.name .. '_net_D.t7', netD:clearState())
          if cmd.end2end then
            texture_model_save()
          end
      end

      parametersD, gradParametersD = nil, nil -- nil them to avoid spiking memory
      parametersG, gradParametersG = nil, nil
      if cmd.end2end then
        make_param_grads_texture_nill()
      end

      
      log(('End of epoch %d / %d \t Time Taken: %.3f'):format(
              epoch, opt.niter, epoch_tm:time().real))
      parametersD, gradParametersD = netD:getParameters() -- reflatten the params and get them
      parametersG, gradParametersG = netG:getParameters()
  end

end
   
 
function test()
  local loss_batch
  local loss_MSE
  opt.net = opt.saveadr .. cmd.continue_segmentation .. '_net_G.t7'
  local net = util.load(opt.net, opt)
  net:evaluate()
  net = util.cudnn(net)
  net:cuda()

  local counter = 0
  local iou_aver = 0
  local loss_MSE = 0
  
  for i = 1, math.min(data_size, opt.ntrain), opt.batchSize do
 
    local input, output, resized_amodal, resized_modal = get_batch()
    local real_ctx = input:cuda()
    local pred_center
    pred_center = net:forward(real_ctx)

    local sum_iou_batch = 0
    
    local loss_batch = 0

    counter = counter + 1
    if opt.nc_output == 1 then 
      for j = 1, opt.batchSize do
        sum_iou_batch = sum_iou_batch + calc_iou(resized_amodal[j], pred_center[j])
      end
      iou_aver = (iou_aver * (counter - 1) + sum_iou_batch / opt.batchSize) / counter
    elseif opt.nc_output == 3 then
      local cri = nn.MSECriterion()
      loss_batch = cri:forward(resized_amodal, pred_center)
      loss_MSE = (loss_MSE * (counter - 1) + loss_batch) / counter
      log('Loss average: ' .. loss_MSE .. ' ----  just this batch: ' .. loss_batch)
    end

    if opt.nc_output == 3 or  cmd.end2end then
      if torch.rand(1)[1] < opt.vis_prob and opt.nc_output==3 then
        ind = 1
        save_everything(real_ctx[ind], modal[ind], amodal[ind], pred_center[ind],real_center[ind], ind)
      end
      if torch.rand(1)[1] < opt.vis_prob and cmd.end2end then
        pred_center = clean_mask_output(pred_center)
        local im = real_ctx:clone()
        output_end2end, gradient_end2end = test_texture(real_ctx[{{},{1,3},{},{}}], real_center, pred_center, modal_predicted)
        ind = 1
        save_everything(im[ind], modal_predicted[ind], pred_center[ind], output_end2end[ind], output[ind], ind)
      end
    end

  end

end

