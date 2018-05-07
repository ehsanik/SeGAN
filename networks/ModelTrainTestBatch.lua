-- This file implements the train and test function for one batch
-- By: Kiana Ehsani

require('networks/SeGANCri2D')
require('models.SegmentorTail')
require('util/VisualizationUtil')

--Constructing The NN model
model={};
model.jointNN = getSeGANNet()

-- Upsample a mask from src_size to dest_size using spatial bilinear upsampling
function upsample_segmentation(mask, src_size, dest_size)
    local upsample=nn.SpatialUpSamplingBilinear({oheight=dest_size, owidth=dest_size})
    local upsampled = upsample:cuda():forward(mask:cuda():reshape(opt.batchSize, 1,src_size, src_size))
    return upsampled
end

-- Upsample a box of a mask to a final size
function upsample(mask, box, final_size)
  if #(#mask) < 2 then
    mask=mask:reshape(config.output_segm_dim, config.output_segm_dim)
  end
  if not final_size then
    final_size = 500
  end
  if (#box)[1] == 5 then
    box = {box[2], box[3], box[4], box[5]}
  end
  local upsample=nn.SpatialUpSamplingBilinear({oheight=box[4] - box[2] + 1, owidth=box[3] - box[1] + 1}) 
  local upsampled = upsample:cuda():forward(mask:cuda():reshape(opt.batchSize, 1,config.output_segm_dim, config.output_segm_dim))
  local big = torch.Tensor(final_size, final_size):zero():cuda()
  big[{{box[2], box[4]},{box[1], box[3]}}] = upsampled:squeeze():cuda()
  return big
end

-- Computing the learning rate based on the current iteration and the learning rate decay rules defined in data_settings.lua
function model:LearningRateComp(iter)
  local lIter = (iter % config.nResetLR)+1;
  local regimes= config.regimes;
  for _, row in ipairs(regimes) do
    if lIter >= row[1] and lIter <= row[2] then
      return row[3];
    end
  end
  return regimes[#regimes][3]  
end

-- Apply the weights for SV, SI, SF based on the segmenation mask. 
function get_weights(mask)
  local cleaned = clean_mask(mask)
  local forone = torch.mul(cleaned, config.weights[2]) -- if it is one
  local forzero = torch.add(cleaned, -1)
  forzero = torch.mul(forzero, - config.weights[1]) --if it is zero
  forone:add(forzero) 
  return forone
end


-- Train or Test one batch
-- @param train True or False, if True training phase, if False testing phase
-- @param input Input Image
-- @param amodal SF groundtruth mask
-- @param modal SV groundtruth mask
-- @param gradient_end2end The gradient from the texture generation network that needs to be backpropagated towards the segmentation network
-- @param box_target the bounding box for the object
-- @param predicted the SV predicted mask 
function model:TrainOneBatch(input, amodal, modal, train, gradient_end2end, box_target, predicted)

  -- Set into training phase (just active the dropouts)
  amodal_resized = amodal
  modal_resized = modal
  modal_predicted = predicted
  

  if train == true then
    model.fullNN:training();
  else
    model.fullNN:evaluate();
  end


  -- Forward passs
  local tab
  if config.BoxTask then 
    model.fullNN:forward(input)
  else
    tab = {input,box_target}
    model.fullNN:forward(tab)
  end

  local amodal_iou = 0
  local modal_iou = 0
  local loss = 0


  -- calulcating the intersection over unions for different cases
  for i = 1,config.batchSize do
    amodal_iou = amodal_iou + calc_iou(amodal_resized[i], model.fullNN.output[i])
    modal_iou = modal_iou + calc_iou(modal_resized[i], model.fullNN.output[i])
  end
  amodal_iou = amodal_iou / opt.batchSize
  modal_iou = modal_iou / opt.batchSize

  
  -- Calculating loss for training
  if  train == true then 
    local weights = get_mask(modal_resized:reshape(config.batchSize, config.output_segm_dim, config.output_segm_dim), amodal_resized:reshape(config.batchSize, config.output_segm_dim, config.output_segm_dim))
    model.criterion =  nn.KianaCriterion2D(weights, false):cuda() --size average false by default
    loss = model.criterion:forward(model.fullNN.output:reshape(config.batchSize, config.output_segm_dim, config.output_segm_dim), amodal_resized:reshape(config.batchSize, config.output_segm_dim, config.output_segm_dim))
  end

  if not overall_loss then
    overall_loss = 0
    TrainOneBatch_counter = 0
  end

  if not iou_modal_all then
    iou_modal_all = 0
    iou_amodal_all = 0
  end

  TrainOneBatch_counter = TrainOneBatch_counter + 1
  overall_loss = overall_loss + loss
    
  -- updating the overall iou metric
  iou_amodal_all = iou_amodal_all + amodal_iou
  iou_modal_all = iou_modal_all + modal_iou
  
  if train == true then
    model.fullNN:zeroGradParameters()
    amodal_resized = amodal_resized:cuda()
    local bwCri = model.criterion:backward(model.fullNN.output,amodal_resized)
    local downsampled_gradient = torch.Tensor(#bwCri):cuda()
    gradient_end2end=gradient_end2end:float()
    for ii = 1, opt.batchSize do
      local tmp = image.scale(gradient_end2end[ii], cmd.output_segm_dim, cmd.output_segm_dim) 
      downsampled_gradient[ii] = tmp
    end 

    --backward and update parameters
    model.fullNN:backward(tab,bwCri + downsampled_gradient:mul(opt.lambdac))
    model.fullNN:updateParameters(model.learningRate)

  end

  if not train then
    calc_SV_SI(modal_resized, amodal_resized, model.fullNN.output, 'output')
  end


  log(('LR %18.18f ---- average_loss %f ---- aver_modal %f ---- aver_amodal %f'):format(
                      model.learningRate,
                      overall_loss / TrainOneBatch_counter, 
                      iou_modal_all / TrainOneBatch_counter,
                      iou_amodal_all / TrainOneBatch_counter
       ));
  return model.fullNN.output
end

-- The function for forwarding the input with a cropping box
-- Used in the GAN input generation
function just_forward(input, box_target)
  local tab = {input,box_target} 
  model.fullNN:forward(tab)
  return model.fullNN.output
end











































































function model:LoadCaffeImageNN(caffeFilePath)
  local protoFile = caffeFilePath.proto
  local modelFile = caffeFilePath.model
  local meanFile  = caffeFilePath.mean

  require 'loadcaffe'
  local caffeModel = loadcaffe.load(protoFile,modelFile,'cudnn');
  caffeModel:remove(24);
  caffeModel:remove(23);
  caffeModel:remove(22);
  local caffeParams = GetNNParamsToCPU(caffeModel);

  --local nAdditionalChn = GetValuesSum(GetEnableInputTypes(config.input_data)) - 3; -- minus original 3 for rgb
  --local firstLayerRandom = torch.randn(96, 3, 11, 11):float()

  --caffeParams[1] = torch.cat(caffeParams[1],firstLayerRandom, 2)

  --local hh  = torch.cat(caffeParams[1],caffeParams[1], 2)
  --caffeParams[1] = torch.cat(caffeParams[1],hh, 2)
  caffeParams[1] = torch.cat(caffeParams[1],caffeParams[1], 2)

  
  LoadNNlParams(model.imageNN, caffeParams);
  model.jointNN:apply(rand_initialize)
  LoadCaffeMeanStd(meanFile);
end


function model:SaveModel(fileName)
  -- local saveModel ={};
  -- -- reading model parameters to CPU
  -- saveModel.imageNN     = GetNNParamsToCPU(model.imageNN);
  -- saveModel.jointNN     = GetNNParamsToCPU(model.jointNN);
  -- -- saving into the file
  -- torch.save(fileName,saveModel)
  torch.save(fileName,model)
end


function model:LoadModel(fileName)
  if fileName=="resnet" then
      log('WARNING: specified initialized model is ignored because resnet model will be loaded! ') 
      local resnet = torch.load(config.resnet_path)

      if cmd.NN then
        model.fullNN = resnet:cuda()
        return 
      end

      if not cmd.NN then
        resnet:remove(11)
        resnet:remove(10)  --in hamun layer koofti view e
      end
      -- resnet:remove(9)
      -- removed this
      
      -- I changed here from 6 to 4
      local tmpLayer = cudnn.SpatialConvolution(4,64,7,7,2,2,3,3)
      local inputLayer = resnet:get(1)
      tmpLayer.weight[{{},{1,3},{},{}}]:copy(inputLayer.weight)
      -- I changed the follwing 
      tmpLayer.weight[{{},{4},{},{}}]:copy(inputLayer.weight[{{},{1}}])
      --tmpLayer.weight[{{},{4,6},{},{}}] = torch.randn(64, 3, 7, 7):float()
      tmpLayer.bias:copy(inputLayer.bias) 
      resnet:remove(1)
      resnet:insert(tmpLayer,1)

      model.imageNN = resnet

  
  end

  -- model.fullNN = nn.Sequential():add(model.imageNN):add(model.jointNN)

  -- if cmd.NN then 
  --   model.fullNN  = model.imageNN
  -- else
    model.fullNN = getROINet(model.imageNN, model.jointNN)
  model.fullNN:cuda();

  -- model:SaveModel(config.logDirectory .. '/init.t7') 
end

