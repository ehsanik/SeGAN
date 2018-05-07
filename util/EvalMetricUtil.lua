-- Utility functions for calculating the Intersection over Union for evaluation
-- By: Kiana Ehsani

function clean_mask_output(result)
  if torch.sum(result) == 0 then
    return result:float()
  end
  local thr = 0.5
  return  torch.ge(result,thr):float() 
end

function calc_iou(gt, result)
  local output = clean_mask_output(result)
  local clean_gt = clean_mask_output(gt)
  local sum = clean_gt
  sum:add(output)
  local intersection = torch.ge(sum, 2):float()
  local union = torch.ge(sum, 1):float()
  local iou = 0
  if torch.sum(union) ~= 0 then
    iou = torch.sum(intersection) / torch.sum(union)
  end
  return iou
end

-- Calulate the IOU for just the SI mask for evaluation
function calcIouJustAmodal(outputi, modali, amodali)
  local out = clean_mask(outputi)
  local amod = clean_mask(amodali)
  local mod = clean_mask(modali)
  local res = torch.csub(out, mod)
  res = torch.gt(res, 0):float()
  local target = torch.csub(amod, mod)
  target = torch.gt(target, 0):float()
  if torch.sum(res) == 0 and torch.sum(target) == 0 then
    return 1
  end
  return calc_iou(target, res)
end


-- Calculate the SI based on the SV and SF from groundtruth and the predicted output
-- @param modal_resized groundtruth SV
-- @param amodal_resized groundtruth SF
-- @param output predicted SF
-- @param tag Different tags in case we are calculating the IOU for different models
function calc_SV_SI(modal_resized, amodal_resized, output, tag)
    local output = clean_mask_output(output):squeeze()
    local weights = get_mask(modal_resized, amodal_resized)
    local justSIoutput = output:clone():cuda()
    justSIoutput[torch.ne(weights, 2)] = 0
    local justSIgt = amodal_resized:clone():cuda()
    justSIgt[torch.ne(weights, 2)] = 0
    local justSIIOU = calc_iou(justSIgt, justSIoutput)
    if tag == 'output' then ind = 1 else ind = 2 end
    if not just_iou_SI then
        just_iou_SI = {0,0}
        counter_SI_iou = 0
    end
    counter_SI_iou = counter_SI_iou + 1
    just_iou_SI[ind] = just_iou_SI[ind] + justSIIOU 
    print('Average IOU for SI ' .. just_iou_SI[ind] / counter_SI_iou)

end