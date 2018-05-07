-- Binary cross entropy loss function for SeGAN 
-- By: Kiana Ehsani
local THNN = require 'nn.THNN'
local KianaCriterion2D, parent = torch.class('nn.KianaCriterion2D', 'nn.Criterion')

function KianaCriterion2D:__init(weights, sizeAverage)
    parent.__init(self)
    if weights == nil then
        print('weights are not defined')
        return nil
    end
    self.weights_loss2D = opt.weights_loss2D
    self.norms = opt.norm2D_cvpr
    self.weights = weights:clone()
    if sizeAverage ~= nil then
        self.sizeAverage = sizeAverage
    else
        self.sizeAverage = false
    end
end

function AddDimensions(input, target, weights)
    local selfinput = torch.Tensor(#(weights)):zero():cuda()
    selfinput = selfinput:repeatTensor(3,1,1,1,1):zero():cuda()
    local selftarget = torch.Tensor(#(weights)):zero():cuda()
    selftarget = selftarget:repeatTensor(3,1,1,1,1):zero():cuda()
    for i = 1,3 do
        selfinput[i] = input:clone()
        selfinput[i][torch.ne(weights, i - 1)] = 0
        selftarget[i] = target:clone()
        selftarget[i][torch.ne(weights, i - 1)] = 0
    end
   return selfinput, selftarget
end

function KianaCriterion2D:updateOutput(input, target)

   self.totalOutput = {}
   local res = 0
   self.input , self.target = AddDimensions(input, target, self.weights)

   for i = 1, 3 do
       input = self.input[i]
       target = self.target[i] 
       self.output_tensor = self.output_tensor or input.new(1)
       input.THNN.BCECriterion_updateOutput(
          input:cdata(),
          target:cdata(),
          self.output_tensor:cdata(),
          self.sizeAverage,
          THNN.optionalTensor(nil)
       )
       self.output = self.output_tensor[1]
       self.totalOutput[i] = self.output
       this_weight = self.weights_loss2D[i] / self.norms[i]
       res = res + this_weight * self.totalOutput[i]
   end
   
   return res
end

function KianaCriterion2D:updateGradInput(input, target)
   self.totalGrad = torch.Tensor(#input):zero():cuda()
   self.input , self.target = AddDimensions(input, target, self.weights)
   for i = 1,3 do
      input.THNN.BCECriterion_updateGradInput(
          input:cdata(),
          target:cdata(),
          self.gradInput:cdata(),
          self.sizeAverage,
          THNN.optionalTensor(nil)
       )
 
      this_weight = self.weights_loss2D[i] / self.norms[i]
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