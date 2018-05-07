-- Read a batch from dataset
function get_batch()
  if cmd.istrain ~= 'train' then
    config.train.save_dir = config.train.save_dir:gsub('train', 'test')
  end
  if cmd.cvpr then
    images, resized_full, amodal_cvpr, modal_cvpr , crop_box, original_images, resized_amodal, resized_modal    = GetAnImageBatch(config.train)  
    return images, resized_full, amodal_cvpr, modal_cvpr , crop_box, original_images, resized_amodal, resized_modal 
  end 
  local images, resized_full, resized_amodal, resized_modal, crop_box, original_images  = GetAnImageBatch(config.train)  
  if cmd.no_masks then
    local weights = get_mask(resized_modal, resized_amodal)
    for ch=1,3 do
      if cmd.old_no_masks then
          config.red = config.blue
      end
      (images[{{},{ch},{},{}}])[torch.eq(weights, 2)] = config.red[ch];
      (images[{{},{ch},{},{}}])[torch.eq(weights, 0)] = config.blue[ch]
    end
  end
  return images, resized_full, resized_amodal, resized_modal, crop_box, original_images
end