-- Main function for training/testing the SeGAN network
-- By: Kiana Ehsani

require 'torch'
require 'nn'
require 'optim'
require 'image'
require 'models/models'
require 'gnuplot'
require 'cunn'
require 'cudnn'
require 'xlua'
require 'math'
require 'cunn'
require 'cudnn'
require 'gnuplot'
require 'sys'
require 'image'
require('ROI.ROI')
require('ROI.ROIPooling')
require('networks/SeGANCri')
pl = require'pl.import_into'()
debugger = require 'fb.debugger'
util = paths.dofile('util/util.lua')
paths.dofile('util/TagMapDict.lua')
paths.dofile('SettingsParser.lua');
IOUtilFunctions = paths.dofile('util/IOUtilFunctions.lua')
paths.dofile('util/EvalMetricUtil.lua')
paths.dofile('util/BatchGeneration.lua')



for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)


--local variables

local input_nc = opt.input_nc
local output_nc = opt.output_nc
local idx_A = nil
local idx_B = nil

local ndf = opt.ndf
local ngf = opt.ngf
local real_label = 1
local fake_label = 0

----------------------------------------------------------------------------

local real_A = torch.Tensor(opt.batchSize, input_nc, opt.fineSize, opt.fineSize)
local real_B = torch.Tensor(opt.batchSize, output_nc, opt.fineSize, opt.fineSize)
local fake_B = torch.Tensor(opt.batchSize, output_nc, opt.fineSize, opt.fineSize)
local real_AB = torch.Tensor(opt.batchSize, output_nc + input_nc*opt.condition_GAN, opt.fineSize, opt.fineSize)
local fake_AB = torch.Tensor(opt.batchSize, output_nc + input_nc*opt.condition_GAN, opt.fineSize, opt.fineSize)
local errD, errG, errL1 = 0, 0, 0
local epoch_tm = torch.Timer()
local tm = torch.Timer()
local data_tm = torch.Timer()
local df_do_AE, df_do, df_dg
local netG
local netD
local criterionAE, criterion, optimStateG, optimStateD
local parametersD, gradParametersD
local parametersG, gradParametersG
local output_end2end, gradient_end2end
local input_critical, output_critical, resized_amodal_critical, resized_modal_critical, crop_box, original_images
----------------------------------------------------------------------------


local function createRealFake() 
    -- load real
    data_tm:reset(); data_tm:resume()
    
    input_critical, output_critical, resized_amodal_critical, resized_modal_critical, crop_box, original_images, resized_amodal_texture, resized_modal_texture = get_batch() 

    weight_mask = get_mask(resized_modal_critical, resized_amodal_critical)
    data_tm:stop()
    
    if cmd.cvpr then
      real_A = original_images --not copying
      box_A = crop_box
      real_B = resized_amodal_critical
    else
      real_A:copy(input_critical)
      if opt.nc_output == 1 then
        real_B:copy(resized_amodal_critical)
      else
        real_B:copy(output_critical)
      end
    end

    if not cmd.cvpr then
      if opt.condition_GAN==1 then
        real_AB = torch.cat(real_A,real_B,2)
      else
        real_AB = real_B -- unconditional GAN, only penalizes structure in B
      end
    end
      
    -- create fake
    if cmd.cvpr then
      if not cmd.NN then
        fake_B = just_forward(real_A, box_A)
      end
    else
      fake_B = netG:forward(real_A)
    end
    
    if not cmd.cvpr then
      if opt.condition_GAN==1 then
          fake_AB = torch.cat(real_A,fake_B,2)
      else
          fake_AB = fake_B -- unconditional GAN, only penalizes structure in B
      end
    end
end

-- Train the network end 2 end for opt.niter number of epochs
local function train_end2end()

  local l2 = nn.MSECriterion():cuda()
  local l1 = nn.AbsCriterion():cuda()
  local counter = 1
  epoch_texture = 1

  local input_features_NN, output_features_NN, dataset_files
  dataset_files = {}
  

  for epoch = 1, opt.niter do
    epoch_texture = epoch
    epoch_tm:reset()
    for i = 1, math.min(data_size, opt.ntrain), opt.batchSize do
        --from train function inside the for
        model.learningRate = model:LearningRateComp(counter);
        tm:reset()
        createRealFake() 
        
        if cmd.end2end then
          local amodal_pred = clean_mask_output(upsample_segmentation(fake_B, config.output_segm_dim, opt.fineSize))
          
          output_end2end, gradient_end2end = train_texture(input_critical[{{},{1,3},{},{}}], output_critical, amodal_pred, modal_predicted) 

        else
          gradient_end2end = torch.Tensor(opt.batchSize, 1, opt.fineSize, opt.fineSize):zero():cuda()
        end

        
        model:TrainOneBatch( real_A,real_B, resized_modal_critical, true, gradient_end2end, box_A, modal_predicted_cvpr);
        

        counter = counter + 1

        if counter % opt.print_freq == 0 then
            local curItInBatch = ((i-1) / opt.batchSize)
            local totalItInBatch = math.floor(math.min(data_size, opt.ntrain) / opt.batchSize)
            log(('Epoch: [%d][%8d / %8d]\t Time: %.3f  DataTime: %.3f  '
                    )
                    :format(
                     epoch, curItInBatch, totalItInBatch,
                     tm:time().real / opt.batchSize, data_tm:time().real / opt.batchSize
                     ))              
        end
        collectgarbage('collect')

    end
    
    -- Saving the models
    if epoch % opt.save_epoch_freq == 0 then
        print('saving')
        local fileName = opt.saveadr .. opt.name .. '_' ..'Model.t7';
        log('Saving NN model in ----> ' .. fileName .. '\n');
        model:SaveModel(fileName);
        -- Saving backup for data overflow issues in the server
        fileName = opt.saveadr .. opt.name .. '_' ..'Model_backup.t7';
        log('Saving NN model in ----> ' .. fileName .. '\n');
        model:SaveModel(fileName);
        flushit()
        if cmd.end2end then
          texture_model_save()
        end
    end

    -- Logging the report and emptying the param grads for texture network
    log(('End of epoch %d / %d \t Time Taken: %.3f'):format(
            epoch, opt.niter, epoch_tm:time().real))
    if cmd.end2end then
      make_param_grads_texture_nill()
    end
    
  end

end


--Test the end2end network
local function test_end2end()

  local l2 = nn.MSECriterion():cuda()
  local l1 = nn.AbsCriterion():cuda()
  local counter = 1
  epoch_texture = 1

  local input_features_NN, output_features_NN, dataset_files
  dataset_files = {}
  
  epoch_texture = epoch
  epoch_tm:reset()
  for i = 1, math.min(data_size, opt.ntrain), opt.batchSize do
      --from train function inside the for
      model.learningRate = model:LearningRateComp(counter);
      tm:reset()
      createRealFake() 
      
      if cmd.end2end then
        local amodal_pred = clean_mask_output(upsample_segmentation(fake_B, config.output_segm_dim, opt.fineSize))
        
        if cmd.realdata then 
          output_end2end, gradient_end2end = test_texture(input_critical[{{},{1,3},{},{}}], output_critical, resized_amodal_texture, resized_modal_texture)   
        else
          output_end2end, gradient_end2end = test_texture(input_critical[{{},{1,3},{},{}}], output_critical, resized_amodal_texture, modal_predicted)   
        end
      
      else
        gradient_end2end = torch.Tensor(opt.batchSize, 1, opt.fineSize, opt.fineSize):zero():cuda()
      end

      fake_B = model:TrainOneBatch( real_A,real_B, resized_modal_critical, false, gradient_end2end, box_A, modal_predicted_cvpr);

      if cmd.end2end then
        loss_batch = l2:forward(output_end2end, output_critical)
        l1_batch = l1:forward(output_end2end, output_critical)
        if loss_average then
          loss_average = loss_average + loss_batch
          loss_l1_aver  = loss_l1_aver + l1_batch
        else
          loss_average = loss_batch
          loss_l1_aver = l1_batch
        end
        log('For texture task: L2 average: ' .. loss_average / counter .. ' L1 average: ' .. loss_l1_aver / counter)
        
      end

      counter = counter + 1

      if counter % opt.print_freq == 0 then
          local curItInBatch = ((i-1) / opt.batchSize)
          local totalItInBatch = math.floor(math.min(data_size, opt.ntrain) / opt.batchSize)
          log(('Epoch: [%8d / %8d]\t Time: %.3f  DataTime: %.3f  '
                  )
                  :format(
                    curItInBatch, totalItInBatch,
                    tm:time().real / opt.batchSize, data_tm:time().real / opt.batchSize
                  ))              
      end
      collectgarbage('collect')
      
  end -- for i = 1, math.min(data_size, opt.ntrain), opt.batchSize
  
end



if cmd.end2end then
  paths.dofile('networks/End2EndNetwork.lua')
end



if not cmd.cvpr then

  paths.dofile('networks/GANTextureNet.lua')
  constructorTextureNet()

  if opt.istrain == 'train' then
    train()
  else
    test()
  end

else

  paths.dofile('networks/ModelTrainTestBatch.lua')

  if cmd.NN then
    model:LoadModel("resnet");
    log(model.fullNN);
  elseif opt.continue_train == 1 or opt.istrain ~= 'train'then
    local loadedModel = torch.load(cmd.reload .. '_Model.t7')
    model.fullNN = loadedModel.fullNN
    print('Network is loaded') 
  else
    model:LoadModel("resnet")
  end


  if opt.istrain == 'train' then
    train_end2end()
  else
    test_end2end()
  end

end