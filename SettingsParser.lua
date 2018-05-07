-- Settings and command line parser

local weight_loss_delta = {1, 5, 3}

cmd = torch.CmdLine()
cmd:option('-delta1', weight_loss_delta[1], 'lambda_SI')
cmd:option('-delta2', weight_loss_delta[2], 'lambda_SV')
cmd:option('-delta3', weight_loss_delta[3], 'lambda_SF')
cmd:option('-exprate', 0.2, 'expand ratio for the bounding box for SV')
cmd:option('-istrain', 'train', 'train or test or val')
cmd:option('-postfix', '', 'postfix for saving models and logs') 
cmd:option('-category', '', 'if you wanna train for specific categories')
cmd:option('-weights_segmentation', '', 'address of weights for segmentation network')
cmd:option('-weights_texture', '', 'address of weights for texture network')
cmd:option('-predictedSV', false, 'predicted masks from multipath or ground truth SV')
cmd:option('-output_nc', 3, 'Number of channels of the output')
cmd:option('-amodal_not_given', true, 'set this if you want to use ground truth amodal for texture generation') 
cmd:option('-onebatch', false, 'one batch training for sanity check')
cmd:option('-batchSize', 1, 'batchSize')
cmd:option('-lambda', 100, 'L1 weight')
cmd:option('-lambda2', 1, 'texture weight')
cmd:option('-lambdac', .1, 'CVPR weight') 
cmd:option('-both_masks', false, 'give SV and SF as input to texture network')
cmd:option('-no_masks', false, 'give no mask as input to texture network')
cmd:option('-old_no_masks', false, 'give no mask as input to texture network but no change in other parts of the network')
cmd:option('-half_both_masks', false, 'give SV and SI as input to texture network')
cmd:option('-vis_prob',0, 'if you want to randomly visualize set to the ratio you wish to save visualization for')
cmd:option('-netG', 'unet', 'unet or encoder_decoder')
cmd:option('-lr', 0.0002, 'learning rate') 
cmd:option('-condition_GAN', 1, 'do you want conditional GAN or not')
cmd:option('-nobg', false, 'no background') 
cmd:option('-end2end', false, 'train end to end, if not train each part individually') 
cmd:option('-output_segm_dim', 58, 'size of final segmentation before upsampling') 
cmd:option('-baseLR', 1e-3, 'starting learning rate for cvpr model')
cmd:option('-realdata', false, 'natural images dataset (A subset of Pascal)')
cmd:option('-save_output', '', 'where to save output in case of visualization')
cmd:option('-NN', false, 'nearest neighbor model')
cmd:option('-occl_thr_vis', 1, 'occlusion rate threshold for visualization')
cmd:option('-l1_thr', 100, 'l1 loss threshold for visualization')
cmd:option('-here', false, 'read from current directory instead of dataset')



cmd = cmd:parse(arg)
cmd.loadSize = 256 
cmd.fineSize = 256
cmd.loadSize_cvpr = 256
cmd.fineSize_cvpr = 256
cmd.multipath = cmd.predictedSV
cmd.continue_texture = cmd.weights_texture
cmd.reload = cmd.weights_segmentation
cmd.maxVis = 100
cmd.cvpr = true
 

if cmd.end2end then
  cmd.nobg = true
end

if cmd.cvpr then
  cmd.output_nc = 1
  cmd.amodal_not_given = true
  cmd.exprate = cmd.exprate 
  cmd.loadSize_cvpr = cmd.output_segm_dim
  cmd.fineSize_cvpr = cmd.output_segm_dim
  cmd.postfix = '_cvpr' .. cmd.postfix
  cmd.postfix = '_baseLRcvpr' .. cmd.baseLR  .. cmd.postfix
end

if cmd.end2end then
  cmd.postfix = '_end2end' .. cmd.postfix
end

if cmd.multipath then
  cmd.postfix = '_multipath' .. cmd.postfix
end

if cmd.old_no_masks then
  if not cmd.cvpr then
    cmd.no_masks = true
  else
    cmd.no_masks = false
  end
  cmd.postfix = '_old_nomask' .. cmd.postfix
end
cmd.random_jitter = true
if cmd.onebatch or cmd.istrain ~= 'train' then
  cmd.random_jitter = false
end
if cmd.istrain ~= 'train' then
  cmd.batchSize = 1
  cmd.continue_train = 1
end

if cmd.continue_texture ~= '' then
  cmd.continue_train = 1
end

if cmd.lr ~= 0.0002 then
  cmd.postfix = '_lr' .. cmd.lr .. cmd.postfix
end
if cmd.condition_GAN ~= 1 then
  cmd.postfix = '_no_cond_GAN' .. cmd.postfix
end

if cmd.realdata then
  cmd.postfix = '_realdata' .. cmd.postfix
end

cmd.input_nc = 4
if cmd.half_both_masks then
  cmd.both_masks = true
end
if cmd.no_masks then
  cmd.input_nc = 3
  cmd.amodal_not_given = true
  cmd.postfix = '_nomask' .. cmd.postfix
end
if cmd.both_masks then
  cmd.input_nc = 5
  cmd.amodal_not_given = true
  cmd.postfix = '_bothmasks' .. cmd.postfix
end

if cmd.netG == 'encoder_decoder' then
  cmd.postfix = '_encdec' .. cmd.postfix
end

if cmd.lambda ~= 100 then
  cmd.postfix = '_lambda' .. cmd.lambda .. cmd.postfix

end
cmd.postfix = '_lambdatwo' .. cmd.lambda2 .. cmd.postfix
cmd.postfix = '_lambdac' .. cmd.lambdac .. cmd.postfix
exprate = cmd.exprate
weight_loss_delta = {cmd.delta1, cmd.delta2, cmd.delta3}

if cmd.amodal_not_given then
  cmd.amodal_given = false
else
  cmd.amodal_given = true
end

if cmd.category ~= '' then
  cmd.postfix = '_' .. cmd.category .. cmd.postfix
end
SERVER = cmd.server

opt = {
  norm2D_cvpr = {37000,27000,12000},
  nobg = cmd.nobg,
  random_jitter = cmd.random_jitter,
  expand_jitter_ratio = 0.1,
  clean = true,
  category = cmd.category,
  myloss = cmd.myloss,
  weights_loss = weight_loss_delta,
  istrain = cmd.istrain, 
  postfix =  cmd.postfix,
  amodal_given = cmd.amodal_given,
  wtl2 = 0.999,               -- 0 means don't use else use with this weight
  onebatch = cmd.onebatch,
  expand_ratio = exprate, 
  crop_ratio = exprate, 
  vis_prob = cmd.vis_prob,
  output_nc = cmd.output_nc,         
  batchSize = cmd.batchSize,          -- # images in batch
  loadSize = cmd.loadSize,   -- scale images to this size for texture generation network
  fineSize = cmd.fineSize,         --  then crop to this size for texture generation network
  loadSize_cvpr = cmd.loadSize_cvpr,  -- scale images to this size for segmentation network
  fineSize_cvpr = cmd.fineSize_cvpr,         --  then crop to this size for segmentation network
  ngf = 64,               -- #  of gen filters in first conv layer
  ndf = 64,               -- #  of discrim filters in first conv layer
  input_nc = cmd.input_nc,           -- #  of input image channels
  newCache = 'newdata_read500_fullmeansub_recover', -- address of where to save the input images cache to speed up the data loading process
  niter = 200,            -- #  of iter at starting learning rate
  lr = cmd.lr,            -- initial learning rate for adam
  beta1 = 0.5,            -- momentum term of adam
  ntrain = math.huge,     -- #  of examples per epoch. math.huge for full dataset
  flip = 1,               -- if flip the images for data argumentation
  gpu = 1,                -- gpu = 0 is CPU mode. gpu=X is GPU mode on GPU X
  name = 'SeGAN',              -- name of the experiment, should generally be passed on the command line
  nThreads = 1,                -- # threads for loading data
  save_epoch_freq = 1,        -- save a model every save_epoch_freq epochs (does not overwrite previously saved models)
  save_latest_freq = 1000000000,     -- save the latest model every latest_freq sgd iterations (overwrites the previous latest model)
  print_freq = 1,             -- print the debug information every print_freq iterations
  display_freq = 100,          -- display the current results every display_freq iterations
  save_display_freq = 5000,    -- save the current display of results every save_display_freq_iterations
  continue_train=0,            -- if continue training, load the latest model: 1: true, 0: false
  serial_batches = 0,          -- if 1, takes images in order to make batches, otherwise takes them randomly
  serial_batch_iter = 1,       -- iter into serial image list
  checkpoints_dir = './checkpoints', -- models are saved here
  cudnn = 1,                         -- set to 0 to not use cudnn
  condition_GAN = cmd.condition_GAN,                 -- set to 0 to use unconditional discriminator
  use_GAN = 1,                       -- set to 0 to turn off GAN term
  use_L1 = 1,                        -- set to 0 to turn off L1 term
  which_model_netD = 'basic', -- selects model to use for netD
  which_model_netG =  cmd.netG,  -- selects model to use for netG encoder_decoder
  n_layers_D = 0,             -- only used if which_model_netD=='n_layers'
  lambda = cmd.lambda,               -- weight on L1 term in objective
  lambda2= cmd.lambda2,               -- weight on texture adversarial loss
  lambdac= cmd.lambdac,               -- weight on total texture loss for backpropagating into segmentation network
  open_size = 500,
  norm1C = {0.025, 0.01, 0.03}, 
  norm3C = {1,1,1},
  sum = {0,0,0},
  counter = 0, 
  data = 'dataset/train',
}
if cmd.continue_train then
  opt.continue_train = 1
end
opt.weights_loss2D = opt.weights_loss

if cmd.lambda == -1 then
  opt.lambda = 100
  opt.use_GAN = 0
end

opt.DataRootPath = 'dyce_data'
opt.datasetfile = opt.DataRootPath .. '/annotations/'
opt.saveadr = 'cache/'.. opt.name ..'/checkpoints/'


opt.nc_output = opt.output_nc
opt.nc_input = opt.input_nc
if opt.istrain ~= 'train' then
  opt.batchSize = 1 
end

if opt.istrain == 'train' or opt.onebatch then
  opt.datasetfile = opt.datasetfile .. 'train'
elseif opt.istrain == 'val' then
  opt.datasetfile = opt.datasetfile .. 'val' 
  elseif opt.istrain == 'trainvis' then
  opt.datasetfile = opt.datasetfile .. 'full'
  elseif opt.istrain == 'testval' then
  opt.datasetfile = opt.datasetfile .. 'testval' 
else
  opt.datasetfile = opt.datasetfile .. 'test' 
end

if cmd.multipath then
  opt.datasetfile = opt.datasetfile .. '_multipath'
end
if cmd.realdata then
  opt.datasetfile = opt.datasetfile .. '_real'
end

opt.datasetfile = opt.datasetfile .. '.txt'


opt.isBig = ''
if opt.nc_output == 3 then
  opt.isBig = opt.isBig .. '_newtask'
else
  opt.isBig = opt.isBig .. '_prevtask'
end
opt.isBig = opt.isBig .. '_expand_' .. opt.expand_ratio ..'_crop_' .. opt.crop_ratio .. '_newloss_' .. opt.weights_loss[1] .. '_' .. opt.weights_loss[2] .. '_' .. opt.weights_loss[3] 
if opt.amodal_given then
  opt.isBig = opt.isBig .. '_amodalgiven'
end
opt.isBig = opt.isBig .. '_wtl2_' .. opt.wtl2
opt.isBig = opt.isBig .. opt.postfix
if opt.myloss then
  opt.isBig = opt.isBig .. '_myloss'
end

opt.name = opt.name .. opt.isBig
opt.name_texture = opt.name .. "_texture"
opt.clean_mask_thr = cmd.loadSize_cvpr * cmd.loadSize_cvpr / (1000 * 1000), 


paths.mkdir(opt.saveadr )

---- options
 
config={};


amodal_path = 'amodal'
images_path = "Images"
modal_path = 'modal'

if cmd.multipath then
  modal_path = 'multipath_masks'
end
if cmd.realdata then
  modal_path = 'realimage_modal'
  amodal_path = 'gt_amodal_real' 
  images_path = 'realImage'
end


config.baseLR = cmd.baseLR
config.stepLR = 20000
-- Learning rate decay rules
config.regimes = { 
    -- start, end,    LR,
    {  1,     config.stepLR,   config.baseLR, }, 
    { config.stepLR + 1,     config.stepLR * 2,   config.baseLR * 0.1, },
    { config.stepLR * 2 + 1,     config.stepLR * 3,   config.baseLR * 0.001, },
    {config.stepLR * 3 + 1,     config.stepLR * 4,  config.baseLR * 0.0001,},
  };
config.batchSize = opt.batchSize
config.expandratio  = cmd.exprate
config.output_segm_dim = cmd.output_segm_dim
config.DataRootPath = opt.DataRootPath
config.imagemeanK = {0.378279580698929, 0.3383278679013845, 0.2969336519420535} 
config.red = {1,0,0}
config.blue = {0,0,1} 
for i = 1,3 do
  config.red[i] = config.red[i] - config.imagemeanK[i]
  config.blue[i] = config.blue[i] - config.imagemeanK[i]
end
config.input_data = {
  
  a_image = {
    dir       = config.DataRootPath .. '/' .. images_path,--
    nChannels = 3,
    type      = "png",
    suffix    = "",
    mean      = {},
    std       = {},
    enable    = true,
    croppable = true,
  },

  modal_mask = {
    dir       = config.DataRootPath ..'/' .. modal_path,
    nChannels = 1,
    type      = "png",
    suffix    = "",
    mean      = {},
    std       = {},
    enable    = true,
  },  

  amodal_mask = {
    dir       = config.DataRootPath .. '/' .. amodal_path,
    nChannels = 1,
    type      = "png",
    suffix    = "",
    mean      = {},
    std       = {},
    enable    = false,
  },  

  fullobject = {
    dir       = config.DataRootPath .. "/FullObj",
    nChannels = 3,
    type      = "png",
    suffix    = "",
    mean      = {},
    std       = {},
    enable    = false,
  },  
  

}

if cmd.here then
  config.input_data.a_image.dir = './'
  config.input_data.modal_mask.dir = './'
  config.input_data.amodal_mask.dir = './'
  config.input_data.fullobject.dir = './'

end

config.SaveRootPath = config.DataRootPath .. "/logs"
config.CacheRootPath = config.DataRootPath

config.dataset_has_bbx = true
config.nCategories = 39

trainmeta = {
  save_dir = config.CacheRootPath .. "/resnet_train_savedir".. opt.newCache,
  
}
paths.mkdir(trainmeta.save_dir)

testmeta = {
  save_dir = config.CacheRootPath .. "/resnet_test_savedir".. opt.newCache,
  
}

paths.mkdir(testmeta.save_dir)

config.train = trainmeta
config.resnet_path = "resnet-18.t7"
require('models.ROISeGAN')
config.nResetLR = 5000000;

-- save opt
paths.mkdir(opt.checkpoints_dir)
paths.mkdir(opt.checkpoints_dir .. '/' .. opt.name)
file = torch.DiskFile(paths.concat(opt.checkpoints_dir, opt.name, 'opt.txt'), 'w')
file:writeObject(opt)
opt.logfile = file
  
