--[[----------------------------------------------------------------------------
Copyright (c) 2018-present, 
By: Kiana Ehsani
------------------------------------------------------------------------------]]
-- Adding the ROI pooling layer to a Resnet network
function getROINet(resnet, joint)
    
    local resnet_combined = nn.Sequential():add(resnet)
    local shared_roi_info = nn.ParallelTable()
    shared_roi_info:add(resnet_combined)
    shared_roi_info:add(nn.Identity())

    if cmd.NN then return shared_roi_info end

    local ROIPooling = ROIPooling(6,6):setSpatialScale(1/50) 

    local linear = nn.Sequential()
    linear:add(cudnn.SpatialConvolution(512, config.output_segm_dim * config.output_segm_dim, 6, 6, 1, 1, 0, 0, 1))
    linear:add(nn.Reshape(config.output_segm_dim , config.output_segm_dim)) 
    linear:add( nn.Sigmoid() )


    local Netroi = nn.Sequential()
    Netroi:add(shared_roi_info)
    Netroi:add(ROIPooling)
    Netroi:add(linear)
    return Netroi
end