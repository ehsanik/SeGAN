-- Generating the segmentor part of the network
-- By: Kiana Ehsani

function getSeGANNet()

    local SeGANNet = nn.Sequential()
    local concatTab = nn.ConcatTable()
    for i = 1,4 do
        local hole = nn.Sequential()
        hole:add(nn.SpatialDilatedConvolution(512, 1024, 3, 3, 1, 1, i * 6, i * 6,i * 6, i * 6))
        hole:add(cudnn.ReLU(true))
        hole:add(nn.Dropout())
        hole:add(cudnn.SpatialConvolution(1024, 1024, 1, 1, 1, 1, 0, 0, 1))
        hole:add(cudnn.ReLU(true))
        hole:add(nn.Dropout())
        hole:add(cudnn.SpatialConvolution(1024, 512, 1, 1, 1, 1, 0, 0, 1))
        concatTab:add(hole)
        
    end
    SeGANNet:add(concatTab)
    SeGANNet:add(nn.CAddTable())
    
    return SeGANNet

end