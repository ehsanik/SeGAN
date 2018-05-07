-- Utility functions for I/O reading/writing
-- By: Kiana Ehsani
paths.dofile('utilsNewtonian.lua');

-- Initializes the dataset for reading inputs based on the file name
function init_dataset(file_name)

    dataset = lines_from(file_name)
    cur_pointer = 1
    epoch_real = 0
    data_size = #dataset
    new_list = {}
    for i = 1, #dataset do
        data = split_string(dataset[i])
        tag = tagMap[data[8]]
        if cmd.nobg and tag ~= 0 then
            new_list[#(new_list) + 1] = dataset[i]
        end
    end
    if cmd.nobg then
        dataset = new_list
        data_size = #dataset
    end
    print('data_size  ' .. data_size)
end

-- Makes a set from a list
function Set (list)
    local set = {}
    for _, l in ipairs(list) do set[l] = true end
    return set
end

function get_mask(modal, amodal)
  local m = modal:squeeze()
  local a = amodal:squeeze()
  local res = torch.Tensor(#a):fill(0):cuda()
  res[torch.gt(a,0)] = 2
  res[torch.gt(m, 0)] = 1
  local s = #a
  if #s == 2 then
    return res:reshape(1,1,s[1], s[2])
  end
  return res:reshape(s[1], 1, s[2], s[3])
end

-- Cleans the input segmentation mask after resizing, this can just be used for groundtruth masks
function clean_mask_just_reading(result)
  if torch.sum(result) == 0 then
    return result:float()
  end
  local thr = opt.clean_mask_thr 
  return  torch.ge(result,thr):float() 
end

-- Expand the bounding using a rate and given a maximum
-- @param box the bounding box
-- @param rate the ratio the bounding box should be expanded
-- @param max the maximum values it can get
function expand_box(box, rate, max)
  local w = (box[3] - box[1]) * rate
  local h = (box[4] - box[2]) * rate
  local w1, w2, h1, h2
  if opt.random_jitter then
    local jit = opt.expand_jitter_ratio
    w1 = (box[3] - box[1]) * torch.uniform(rate - jit, rate + jit)
    w2 = (box[3] - box[1]) * torch.uniform(rate - jit, rate + jit)
    h1 = (box[4] - box[2]) * torch.uniform(rate - jit, rate + jit)
    h2 = (box[4] - box[2]) * torch.uniform(rate - jit, rate + jit)
  else
    w1 = w
    w2 = w
    h1 = h
    h2 = h
  end
  box = torch.Tensor({box[1] - w1, box[2] - h1, box[3] + w2, box[4] + h2})
  box = torch.floor(box)
  box[1] = math.max(box[1], 1)
  box[2] = math.max(box[2], 1)
  box[3] = math.min(box[3] + 1, max)
  box[4] = math.min(box[4] + 1, max)
  return box
end

-- Obtain a bounding box surrounding an object using its segmentation mask
function get_bbx(mask)
  local clean = clean_mask_just_reading(mask)
  local indices = torch.nonzero(mask:float())
  if mask:mean() == 0 then
    return nil
  end
  local min = torch.min(indices, 1)[1]
  local max = torch.max(indices, 1)[1]
  local box = {min[1], min[2], max[1], max[2]}
  box = torch.Tensor(box)
  box = torch.Tensor({box[2], box[1], box[4], box[3]})
  return box
end

-- Crop a mask using a bounding box and resize it to a target size
-- @param masks the input masks
-- @param box the croping bounding box
-- @param target_size the size of the mask after croping
-- @param base_size the size of the mask before croping
function resize_mask(masks, boxes, target_size, base_size)
    if not target_size then
        target_size = opt.fineSize
    end
    if not base_size then
        base_size = opt.open_size
    end
    local resizeNet = nn.Sequential():add(ROIPooling(target_size,target_size):setSpatialScale(1/2)):cuda() 
    local resizeInput
    resizeInput ={masks, boxes * 2} -- this is for having that 1/2 up there
    if (#(#masks) == 3) then
        resizeInput[1] = masks:reshape(opt.batchSize,1, base_size, base_size) -- this is for having that 1/2 up there
  
    end
    local res = resizeNet:forward(resizeInput)
    return res
end

-- Get the expanding bounding boxes for all the masks in the list 
function get_all_boxes(modal, ratio)
    local boxes = torch.CudaTensor(opt.batchSize, 5)
    for i = 1,opt.batchSize do
        local box = get_bbx(modal[i]:squeeze())
        local max = (#modal)[#(#modal)] --means the last dimension
        if box == nil then
            box = torch.Tensor({1, 1, max, max})
        end
        box = expand_box(box, ratio, max)
        box=torch.Tensor({i, box[1], box[2], box[3], box[4]})
        boxes[{{i},{}}]=box
        
    end
    return boxes
end

function reverse_0_1(im)
    return 1 - im
end

-- Read a batch from the dataset
function GetAnImageBatch(input_cfg)
    if opt.onebatch  then
        cur_pointer = 1
    else
        if cur_pointer == 1 then
            shuffleList(dataset, 0)
        end
    end
    
    local batchSize   = opt.batchSize;

    local target;
    local images;

    local dataset_size = #dataset;

    local all_input_types = GetEnableInputTypes(config.input_data);

    local nChannels = opt.nc_input


    amodal = torch.CudaTensor(batchSize, opt.open_size, opt.open_size)
    modal = torch.CudaTensor(batchSize, opt.open_size, opt.open_size)
    images = torch.CudaTensor(batchSize, nChannels, opt.open_size, opt.open_size);
    fullobj_images = torch.CudaTensor(batchSize, 3, opt.open_size, opt.open_size);
    modal_predicted = torch.CudaTensor(batchSize, opt.open_size, opt.open_size)
    
    filenames = {}
    complete_lines = {}
    box_target = torch.CudaTensor(batchSize, 5)
    
    categoryNum = torch.FloatTensor(batchSize)
    categoryNum:zero()
    
    amodal:zero()
    modal:zero()
    images:zero()
    modal_predicted:zero()
    fullobj_images:zero()

    local imgname = ""

    local cnt = 1

    while cnt <= batchSize do
        local line

        
        if opt.justOneLine then
            line = split_string(opt.justOneLine)
        else
            line    = split_string(dataset[cur_pointer]);
        end
        
        local imid    = line[1]
        local modal_mask   = line[2]
        local amodal_mask   = line[3]
        
        local offset = 0


        

        if config.dataset_has_bbx  and not cmd.realdata then
            local notAligned = torch.Tensor({line[4],line[5], line[6], line[7]})
            notAligned:div(1000 / opt.open_size)
            notAligned = torch.floor(notAligned)
            notAligned = notAligned + 1
            local box = torch.Tensor({cnt, notAligned[1],notAligned[2],notAligned[3],notAligned[4] })

            offset = 4
            box_target[{{cnt},{}}]=box
        end
        local gt_mod
        if cmd.realdata then 
          gt_mod = line[4]
          offset = 1
        else
            gt_mod = amodal_mask:gsub('amodal', 'modal')
        end

        local tag = line[4 + offset]
        -- local fullobj = line[5 + offset] 
        local fullobj
        if not cmd.realdata then
            fullobj   = line[5 + offset]
            if fullobj == 'table' then
                fullobj = line[#line]
            end
        end

        -- this is for distribution
        local class_num = tagMap[tag]
        if not class_num then
            print('Class not found for ' .. tag)
            class_num = 0
        end
        if class_num == 0 then 
            class_num = config.nCategories
        end
        local justonetag = false
        if opt.category ~= '' then
            justonetag = true
        end
        -- case for which we are just training for one category
        if (justonetag and class_num ~= tagMap[opt.category]) or (tag == 'objects') then 
             if cur_pointer == dataset_size then
                cur_pointer = 1;
                epoch_real = epoch_real + 1;
                shuffleList(dataset, 0); 
            else
                cur_pointer = cur_pointer + 1;
            end
        else
            filenames[#filenames + 1] = modal_mask
            complete_lines[#complete_lines + 1] = dataset[cur_pointer]
            categoryNum[cnt] = class_num
            local j = 1

            for input_type, conf in orderedPairs(config.input_data) do
                        
              if type(conf) == 'table' and (conf.enable or conf.enable == false) then

                    if input_type == 'a_image' then

                        local impath     = paths.concat(conf.dir, imid);
                        local cache_path = paths.concat(input_cfg.save_dir, input_type .. '_' .. imid .. '.t7');
                        local im = ReadIndividualImage(impath, cache_path, 'image');   
                        CURIMGFILENAME = imid
                        CURIMGADR = impath
                        images[{{cnt}, {j, j + conf.nChannels-1}, {}, {}}] = im;
                        j = j + conf.nChannels

                    elseif input_type == 'amodal_mask' then
                        local impath  = paths.concat(conf.dir, amodal_mask);
                        local cache_path = paths.concat(input_cfg.save_dir, input_type .. '_' .. amodal_mask .. '.t7')
                        local im = ReadIndividualImage(impath, cache_path, 'amodal')
                        local not_clean = im:clone()
                        im = clean_mask_just_reading(im)
                        if opt.clean then
                            not_clean = im
                        end
                        if cmd.realdata then
                            im = reverse_0_1(im)
                            not_clean = im
                        end
                        local gt_modal_mask
                        local modal_path 
                        if opt.amodal_given then
                            images[{{cnt}, {j, j + conf.nChannels -1}, {}, {}}] = not_clean
                            j = j + conf.nChannels
                        end
                        if cmd.both_masks then
                            images[{{cnt}, {j, j + conf.nChannels -1}, {}, {}}] = not_clean
                            AMODAL_IND = j
                            j = j + conf.nChannels
                        end
                        amodal[{{cnt}, {}, {}}] = im 
                        local gt_modal_dir = conf.dir:gsub('amodal', 'modal')
                        local impath  = paths.concat(gt_modal_dir, gt_mod);
                        local cache_path = paths.concat(input_cfg.save_dir, input_type .. '_gtmodal_' .. gt_mod .. '.t7')
                        local im = ReadIndividualImage(impath, cache_path, 'modal')
                        im = clean_mask_just_reading(im)
                        modal[{{cnt}, {}, {}}] = im

                    elseif input_type == 'modal_mask' then
                        local impath  = paths.concat(conf.dir, modal_mask);
                        local cache_path = paths.concat(input_cfg.save_dir, input_type .. '_' .. modal_mask .. '.t7')
                        local im = ReadIndividualImage(impath, cache_path, 'modal')
                        local not_clean = im:clone()
                        im = clean_mask_just_reading(im)
                        if opt.clean then
                            not_clean = im
                        end
                        if not opt.amodal_given and (not cmd.no_masks  or cmd.cvpr) then
                            images[{{cnt}, {j, j + conf.nChannels -1}, {}, {}}] = not_clean 
                            j = j + conf.nChannels
                        end
                        modal_predicted[{{cnt}, {}, {}}] = im

                    elseif input_type == 'fullobject' then
                        if not cmd.realdata then
                            local impath  = paths.concat(conf.dir, fullobj);
                            local cache_path = paths.concat(input_cfg.save_dir, input_type .. '_' .. fullobj:gsub('/', '_') .. '.t7')
                            local im = ReadIndividualImage(impath, cache_path, 'fullobj')
                            fullobj_images[{{cnt}, {1, 3}, {}, {}}] = im
                        end

                    end -- if input_type

                end -- if type(conf) == 'table' 
            
            end -- for input type , conf

            cnt = cnt + 1
            -- Reset the counter and shuffle the dataset
            if cur_pointer == dataset_size then
                epoch_real = epoch_real + 1;
                cur_pointer = 1;
                shuffleList(dataset, 0); 
            else
                cur_pointer = cur_pointer + 1;
            end
        end      --for if chair
    end
    local resized_amodal
    local original_images
    local amodal_cvpr, modal_cvpr

    local crop_box = get_all_boxes(modal_predicted, opt.crop_ratio) 

    if cmd.cvpr then
        original_images = images:clone()
        amodal_cvpr = amodal:clone()
        modal_cvpr = modal:clone()
        modal_predicted_cvpr = modal_predicted:clone()
        amodal_cvpr = resize_mask(amodal_cvpr, crop_box, opt.loadSize_cvpr, opt.open_size)    
        modal_cvpr = resize_mask(modal_cvpr, crop_box, opt.loadSize_cvpr, opt.open_size)
        modal_predicted_cvpr = resize_mask(modal_predicted_cvpr, crop_box, opt.loadSize_cvpr, opt.open_size)
        amodal_cvpr = clean_mask_just_reading(amodal_cvpr)
        modal_cvpr = clean_mask_just_reading(modal_cvpr)
    end

    images = resize_mask(images, crop_box, opt.loadSize, opt.open_size)
    resized_full = resize_mask(fullobj_images, crop_box, opt.loadSize, opt.open_size)
    amodal = resize_mask(amodal, crop_box, opt.loadSize, opt.open_size)
    modal = resize_mask(modal, crop_box, opt.loadSize, opt.open_size)
    modal_predicted = resize_mask(modal_predicted, crop_box, opt.loadSize, opt.open_size)


    if opt.inputisnotequaltooutputdimensionally then
        local boxes = get_all_boxes(modal, opt.expand_ratio) 
        resized_amodal = resize_mask(amodal, boxes, opt.fineSize, opt.loadSize)
        resized_amodal = clean_mask_just_reading(resized_amodal)
        resized_modal = resize_mask(modal, boxes, opt.fineSize, opt.loadSize)
        resized_modal = clean_mask_just_reading(resized_modal)

        resized_full = resize_mask(resized_full, boxes, opt.fineSize, opt.loadSize)
        modal_predicted = do_sth_for_it
    else
        resized_amodal = clean_mask_just_reading(amodal)
        resized_modal = clean_mask_just_reading(modal)
    end
    
    if cmd.delta1 == 0 and cmd.delta2 == 0 then
        repeated_modal = resized_modal:repeatTensor(1,opt.nc_output,1,1)
        repeated_amodal = resized_amodal:repeatTensor(1,opt.nc_output,1,1)
        resized_full[torch.eq(repeated_modal, repeated_amodal)] = 0 

    end
    if cmd.no_masks and not cmd.cvpr then
        local weights = get_mask(resized_modal, resized_amodal)
        for ch=1,3 do
            if cmd.old_no_masks then
                config.red = config.blue
            end
            (images[{{},{ch},{},{}}])[torch.eq(weights, 2)] = config.red[ch];
            (images[{{},{ch},{},{}}])[torch.eq(weights, 0)] = config.blue[ch]
        end
    end

    if cmd.half_both_masks then
        SI = resized_amodal - resized_modal
        images[{{},{AMODAL_IND},{},{}}] = SI
    end
    collectgarbage('collect')
    if cmd.cvpr then
        return images, resized_full, amodal_cvpr, modal_cvpr , crop_box, original_images, resized_amodal, resized_modal    
    end
    return images, resized_full, resized_amodal, resized_modal , crop_box, original_images
  
end 

function ReadIndividualImage(im_path, cache_path, type)
    local im;
    if unexpected_condition then error() end
    
    if paths.filep(cache_path) then
        im = torch.load(cache_path);
    else
        local meanstd = {}
        if type == 'image' then
            meanstd = {
              mean = config.imagemeanK,
              std = { 1, 1, 1},
            }
        elseif type == 'fullobj' then
            meanstd = {
                mean = config.imagemeanK,
                std = { 1, 1, 1},
            }
        else
            meanstd = {
                mean = { 0,0,0},
                std = { 1, 1, 1},
            }
        end
        im = loadImage(im_path, opt.open_size, opt.open_size);

        for i=1,3 do
             im[i]:div(255) 
             im[i]:add(-meanstd.mean[i])
             im[i]:div(meanstd.std[i])
        end
        --debugger.enter()
        torch.save(cache_path, im)
    end
    if type == 'image' or type == 'fullobj' then
        return im
    else
        return im[1]
    end

end

function ReadIndividualForce(mat_path, cache_path)
    local force;
    
    if paths.filep(cache_path) then
        force = torch.load(cache_path);
    else
        force = mattorch.load(mat_path);
    
        torch.save(cache_path, force)
    end
    
    return force
end

init_dataset(opt.datasetfile)
