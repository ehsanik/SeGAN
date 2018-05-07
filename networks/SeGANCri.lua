-- Loss function for the segmentation mask, L1 loss function used in GAN for implementing baselines
-- By: Kiana Ehsani

local SeGANCriterion, parent = torch.class('nn.SeGANCriterion', 'nn.Criterion')

function SeGANCriterion:__init(nchannel, weights, sizeAverage)
    parent.__init(self)
    if weights == nil then
        print('weights are not defined')
        return nil
    end
    if nchannel == nil then
        print('nchannel is not defined')
        return nil
    end
    self.nchannel = nchannel
    if self.nchannel == 3 then
        self.weights = weights:repeatTensor(1,self.nchannel,1,1) -- it means we will repeat the channels because the output is three channels
    else
        self.weights = weights:clone()
    end
    if opt.nc_output == 1 then
        self.norms = opt.norm1C
    else --nc_output = 3
        self.norms = opt.norm3C 
    end

    if sizeAverage ~= nil then
        self.sizeAverage = sizeAverage
    else
        self.sizeAverage = true
    end
end

function convert_them(input, target, weights)
    local selfinput = torch.Tensor(#(weights))
    selfinput = selfinput:repeatTensor(3,1,1,1,1):zero():cuda()
    local selftarget = torch.Tensor(#(weights))
    selftarget = selftarget:repeatTensor(3,1,1,1,1):zero():cuda()
    for i = 1,3 do
        selfinput[i] = input:clone()
        selfinput[i][torch.ne(weights, i - 1)] = 0
        selftarget[i] = target:clone()
        selftarget[i][torch.ne(weights, i - 1)] = 0
    end
   return selfinput, selftarget
end

function SeGANCriterion:updateOutput(input, target)

   self.totalOutput = {}
   local res = 0
   self.input , self.target = convert_them(input, target, self.weights)

   for i = 1, 3 do
       input = self.input[i]
       target = self.target[i] 
       self.output_tensor = self.output_tensor or input.new(1)
       input.THNN.AbsCriterion_updateOutput(
          input:cdata(),
          target:cdata(),
          self.output_tensor:cdata(),
          self.sizeAverage
       )
       self.output = self.output_tensor[1]
       self.totalOutput[i] = self.output
       this_weight = opt.weights_loss[i] / self.norms[i]
       res = res + this_weight * self.totalOutput[i]
       opt.sum[i] = opt.sum[i] + self.output
   end

   opt.counter = opt.counter + 1
   return res
end

function SeGANCriterion:updateGradInput(input, target)
   self.totalGrad = torch.Tensor(#input):zero():cuda()
   self.input , self.target = convert_them(input, target, self.weights)
   for i = 1,3 do
       input.THNN.AbsCriterion_updateGradInput(
          self.input[i]:cdata(),
          self.target[i]:cdata(),
          self.gradInput:cdata(),
          self.sizeAverage
       )
       this_weight = opt.weights_loss[i] / self.norms[i]
       self.totalGrad = self.totalGrad:add(self.gradInput:mul(this_weight))
   end
   return self.totalGrad
end





















































































-- local SeGANCriterion, parent = torch.class('nn.SeGANCriterion', 'nn.Criterion')

-- local eps = 1e-12

-- function SeGANCriterion:_init(weights)
--     parent._init(self)

--     if weights then
--        assert(weights:dim() == 1, "weights input should be 1-D Tensor")
--        self.weights = weights
--     end

-- end

-- function SeGANCriterion:updateOutput(input, target)

--     assert(input:nElement() == target:nElement(), "input and target size mismatch")

--     local weights = self.weights

--     local numerator, denom, common

--     if weights ~= nil and target:dim() ~= 1 then
--           weights = self.weights:view(1, target:size(2)):expandAs(target)
--     end

--     -- compute numerator: 2 * |X n Y|   ; eps for numeric stability
--     common = torch.eq(input, target)  --find logical equivalence between both
--     common:mul(2)
--     numerator = torch.sum(common)

--     -- compute denominator: |X| + |Y|
--     denom = input:nElement() + target:nElement() + eps

--     self.output = numerator/denom

--     return self.output
-- end

-- function SeGANCriterion:updateGradInput(input, target)

--     assert(input:nElement() == target:nElement(), "inputs and target size mismatch")

--     --[[ 
--                                 2 * |X| * |Y|   
--                 Gradient =   ----------------------
--                                 |X|*(|X| + |Y|)^2
--         ]]

--     local weights = self.weights
--     local gradInput = self.gradInput or input.new()
--     local numerator, denom, den_term2, output

--     gradInput:resizeAs(input)

--     if weights ~= nil and target:dim() ~= 1 then
--         weights = self.weights:view(1, target:size(2)):expandAs(target)
--     end

--     if weights ~= nil then
--         gradInput:cmul(weights)
--     end

--     if self.sizeAverage then
--         gradInput:div(target:nElement())
--     end

--     -- compute 2 * |X| * |Y|   
--     numerator = 2 * input:nElement() * target:nElement()

--     -- compute |X|
--     denom = input:nElement()

--     -- compute (|X| + |Y|)
--     den_term2 = input:nElement() + target:nElement()

--     -- compute |X| * (|X| + |Y|)^2
--     denom = denom * (den_term2 * den_term2)

--     -- compute gradients
--     gradInput = numerator / denom

--     self.gradInput = gradInput

--     return self.gradInput
-- end