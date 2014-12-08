package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

type SigmoidalNeuron struct {
	inputSignal float64
	outputSignal float64
}

func NewSigmoidalNeuron() *SigmoidalNeuron{
	return &SigmoidalNeuron{0, 0}
}

func (n *SigmoidalNeuron) Calc(input float64) float64{
	n.inputSignal = input
	n.outputSignal = n.sigmoid(input)
	return n.outputSignal
}

func (n *SigmoidalNeuron) Through(input float64) float64{
	n.outputSignal = input
	n.inputSignal = input
	return input
}

func (n *SigmoidalNeuron) sigmoid(x float64) float64{
	return 1.0 / (1.0 + math.Exp2(-x))
}


type DimError struct {
	What string
}

func (e *DimError) Error() string{
	 return fmt.Sprintf("at %s", e.What)
}


type NeuralNetwork struct {
	inputDim, hiddenDim, outputDim int
	inputNeuron, hiddenNeuron, outputNeuron []SigmoidalNeuron
	wih, who [][]float64
}

func NewRandom2dSlice(x int, y int) [][]float64{
	s := make([][]float64, x)
	for i:=0; i < x; i++{
		s[i] = make([]float64, y)
		for j:=0; j < y; j++{
			s[i][j] = rand.Float64()-0.5
		}
	}
	return s
}

func NewNeuralNetwork(inputNum int, hiddenNum int, outputNum int) *NeuralNetwork {
	wih := NewRandom2dSlice(inputNum+1,hiddenNum)
	who := NewRandom2dSlice(hiddenNum+1,outputNum)
	return &NeuralNetwork{inputNum, hiddenNum, outputNum,
				make([]SigmoidalNeuron, inputNum+1),
				make([]SigmoidalNeuron, hiddenNum+1),
				make([]SigmoidalNeuron, outputNum),
				wih, who}
}

func (nn *NeuralNetwork) calc(inputVec []float64) ([]float64, error){
	if len(inputVec) != nn.inputDim{
		return nil, &DimError{"input error"}
	}

	for i:=0; i<nn.inputDim; i++{
		nn.inputNeuron[i].Calc(inputVec[i])
	}
	nn.inputNeuron[nn.inputDim].Through(1.0)//バイアスニューロ

	var sum float64
	for h:=0; h<nn.hiddenDim; h++{
		sum = 0.0
		for i:=0; i<nn.inputDim+1; i++{
			sum+=nn.inputNeuron[i].outputSignal * nn.wih[i][h]
		}
		nn.hiddenNeuron[h].Calc(sum)
	}
	nn.hiddenNeuron[nn.hiddenDim].Through(1.0)

	for o:=0; o<nn.outputDim; o++{
		sum = 0.0
		for h:=0; h<nn.hiddenDim+1; h++{
			sum+=nn.hiddenNeuron[h].outputSignal * nn.who[h][o]
		}
		nn.outputNeuron[o].Calc(sum)
	}
	outputVec := make([]float64, nn.outputDim)
	for j:=0; j<nn.outputDim; j++{
		outputVec[j] = nn.outputNeuron[j].outputSignal
	}
	return  outputVec, nil
}

func (nn *NeuralNetwork) backpropagation(inputVec []float64, targetVec []float64, learningConst float64, tolerance float64) error{
	if nn.inputDim != len(inputVec) {
		return &DimError{"inputVec dim error"}
	}
	if nn.outputDim != len(targetVec) {
		return &DimError{"targetVec dim error"}
	}
	outputVec, _:= nn.calc(inputVec)

	convergeState := true
	for k:=0; k<nn.outputDim; k++ {
		diff := targetVec[k] - outputVec[k]
		if diff > tolerance || diff < -tolerance{
			convergeState = false
		}
	}
	if convergeState == true {
		return nil
	}

	//backpropagation
	delta_o := make([]float64, nn.outputDim)
	delta_h := make([]float64, nn.hiddenDim)
	tmpPreWho := make([][]float64, nn.hiddenDim+1)
	for i:=0; i < nn.hiddenDim+1; i++{
		tmpPreWho[i] = make([]float64, nn.outputDim+1)
	}

	for k:=0; k<nn.outputDim; k++ {
		delta_o[k] = (targetVec[k]-outputVec[k])*outputVec[k]*(1.0-outputVec[k])
		for j:=0; j<nn.hiddenDim+1;j++{
			tmpPreWho[j][k] = nn.who[j][k]
			nn.who[j][k] += learningConst*delta_o[k]*nn.hiddenNeuron[j].outputSignal
		}
	}

	for j:=0; j<nn.hiddenDim;j++{
		delta_h[j]=0.0
		for k:=0; k<nn.outputDim; k++{
			delta_h[j]+=delta_o[k]*tmpPreWho[j][k]
		}
		delta_h[j] *= nn.hiddenNeuron[j].outputSignal * (1.0-nn.hiddenNeuron[j].outputSignal)
		for i:=0; i<nn.inputDim+1; i++{
			nn.wih[i][j] += learningConst * delta_h[j] * nn.inputNeuron[i].outputSignal
		}
	}
	return nil
}

func main(){
	rand.Seed(time.Now().UnixNano())
	nn := NewNeuralNetwork(1,20,1)

	//Train (y=sin(x))
	for i:=0; i<1000000; i++{
		tmpRandom := math.Pi * rand.Float64()
		nn.backpropagation([]float64{tmpRandom}, []float64{math.Sin(tmpRandom)}, 0.7, 0.000001)
	}

	//Run Test
	fmt.Println("  Ans    :    NN")
	for i:=0; i<20; i++{
		var tmp float64
		tmp = math.Pi * float64(i) / 20.0
		outVec, err := nn.calc([]float64{tmp})
		if(err != nil){
			fmt.Println(err.Error())
			return
		}
		fmt.Printf("%f : %f\n", math.Sin(tmp),(outVec[0]))
	}

}
