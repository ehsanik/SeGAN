-- Util functions for visualization 
-- By: Kiana Ehsani

function write_mask_coco(img, masks, adr)
  local res = img:clone()
  local mask = masks:reshape(1,config.amodalSize, config.amodalSize):double() 
  coco.MaskApi.drawMasks(res, mask, 10) 
  image.save(adr, res)
end

function vis_mask(mask)
  gnuplot.imagesc(mask:squeeze()); gnuplot.figure()
end
 
function get_l2(outputi, modali)
  local out = clean_mask(outputi)
  local mod = clean_mask(modali)
  local res = torch.csub(out, mod)
  res = torch.gt(res, 0):float()
  return res
end

 
function visualize_one(mat)
  gnuplot.imagesc(mat);gnuplot.figure()
end


function visualize(input_vis, output_vis, modal_vis, amodal_vis, t)
  local modal_deep = input_vis[{{t},{4},{},{}}]:reshape(500,500)
  local image = input_vis[{{t},{3},{},{}}]:reshape(500,500)
  gnuplot.imagesc(image);  gnuplot.figure()
  gnuplot.imagesc(modal_deep);  gnuplot.figure()
  gnuplot.imagesc((output_vis[t]):reshape(config.output_segm_dim,config.output_segm_dim));  gnuplot.figure()
  gnuplot.imagesc((amodal_vis[t]):reshape(config.output_segm_dim,config.output_segm_dim));  gnuplot.figure()
  gnuplot.imagesc((modal_vis[t]):reshape(config.output_segm_dim,config.output_segm_dim));  gnuplot.figure()
  gnuplot.imagesc((modal_predicted[t]):reshape(config.output_segm_dim,config.output_segm_dim));  gnuplot.figure()
end


function save_img(imag, file_name, meansub, address)
  img = imag:squeeze():clone()
  if file_name and (string.match(file_name, 'image') or meansub) then
    img[1]:add(config.imagemeanK[1])
    img[2]:add(config.imagemeanK[2])
    img[3]:add(config.imagemeanK[3])
  end
  local path = '../../visualizations' .. opt.name .. '/'
  if address and  address == 'qualitative' then
    path = '../../visualizations_qualitative' .. '/'
  end
  paths.mkdir(path)
  if file_name then
    image.save(path.. file_name ..'.png', img)
  else
    image.save(path ..'salam.png', img)
  end
end

function save_everything(img, modal, amodal, output, fullobj, tag, address)
  if save_counter then
    save_counter = save_counter + 1
  else
    save_counter = 1
  end
  tag = save_counter .. '_' .. tag
  save_img(img[{{1,3},{},{}}], tag .. 'image' , true, address)
  save_img(modal, tag .. 'modal', false, address)
  save_img(amodal, tag .. 'amodal', false, address)
  if cmd.delta1 == 0 and cmd.delta2 == 0 then
    save_img(output, tag ..  'just_output' , true, address)
    local repeated_modal=modal:repeatTensor(1,opt.nc_output,1,1)
    output[torch.eq(repeated_modal, 1)] = fullobj[torch.eq(repeated_modal, 1)]
  end
  save_img(output, tag ..  'output' , true, address)
  save_img(fullobj,tag ..  'output_gt' , true, address)

end 