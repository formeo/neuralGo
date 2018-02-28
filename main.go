package main


import (
	"encoding/csv"
	"errors"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"strconv"
	"time"
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)


// neuralNet содержит всю информацию,
// которая определяет обученную сеть.
type neuralNet struct {
	config  neuralNetConfig
	wHidden *mat.Dense
	bHidden *mat.Dense
	wOut    *mat.Dense
	bOut    *mat.Dense
}

// neuralNetConfig определяет архитектуру
// и параметры обучения нашей сети.
type neuralNetConfig struct {
	inputNeurons  int
	outputNeurons int
	hiddenNeurons int
	numEpochs     int
	learningRate  float64
}

// newNetwork инициализирует новую нейронную сеть.
func newNetwork(config neuralNetConfig) *neuralNet {
	return &neuralNet{config: config}
}

// sigmoid является реализацией сигмоиды,
// используемой для активации.
func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

// sigmoidPrime является реализацией производной
// сигмоиды для обратного распространения.
func sigmoidPrime(x float64) float64 {
	return x * (1.0 - x)
}
// backpropagate завершает метод прямого распространения.
func (nn *neuralNet) backpropagate(x, y, wHidden, bHidden, wOut, bOut, output *mat.Dense) error {

	// Обучаем нашу модель в течение определенного
	// количества эпох, используя обратное распространение.
	for i := 0; i < nn.config.numEpochs; i++ {

		// Завершаем процесс прямого распространения.
		hiddenLayerInput := new(mat.Dense)
		hiddenLayerInput.Mul(x, wHidden)
		addBHidden := func(_, col int, v float64) float64 { return v + bHidden.At(0, col) }
		hiddenLayerInput.Apply(addBHidden, hiddenLayerInput)

		hiddenLayerActivations := new(mat.Dense)
		applySigmoid := func(_, _ int, v float64) float64 { return sigmoid(v) }
		hiddenLayerActivations.Apply(applySigmoid, hiddenLayerInput)

		outputLayerInput := new(mat.Dense)
		outputLayerInput.Mul(hiddenLayerActivations, wOut)
		addBOut := func(_, col int, v float64) float64 { return v + bOut.At(0, col) }
		outputLayerInput.Apply(addBOut, outputLayerInput)
		output.Apply(applySigmoid, outputLayerInput)

		// Завершаем обратное расространение.
		networkError := new(mat.Dense)
		networkError.Sub(y, output)

		slopeOutputLayer := new(mat.Dense)
		applySigmoidPrime := func(_, _ int, v float64) float64 { return sigmoidPrime(v) }
		slopeOutputLayer.Apply(applySigmoidPrime, output)
		slopeHiddenLayer := new(mat.Dense)
		slopeHiddenLayer.Apply(applySigmoidPrime, hiddenLayerActivations)

		dOutput := new(mat.Dense)
		dOutput.MulElem(networkError, slopeOutputLayer)
		errorAtHiddenLayer := new(mat.Dense)
		errorAtHiddenLayer.Mul(dOutput, wOut.T())

		dHiddenLayer := new(mat.Dense)
		dHiddenLayer.MulElem(errorAtHiddenLayer, slopeHiddenLayer)

		// Регулируем параметры.
		wOutAdj := new(mat.Dense)
		wOutAdj.Mul(hiddenLayerActivations.T(), dOutput)
		wOutAdj.Scale(nn.config.learningRate, wOutAdj)
		wOut.Add(wOut, wOutAdj)

		bOutAdj, err := sumAlongAxis(0, dOutput)
		if err != nil {
			return err
		}
		bOutAdj.Scale(nn.config.learningRate, bOutAdj)
		bOut.Add(bOut, bOutAdj)

		wHiddenAdj := new(mat.Dense)
		wHiddenAdj.Mul(x.T(), dHiddenLayer)
		wHiddenAdj.Scale(nn.config.learningRate, wHiddenAdj)
		wHidden.Add(wHidden, wHiddenAdj)

		bHiddenAdj, err := sumAlongAxis(0, dHiddenLayer)
		if err != nil {
			return err
		}
		bHiddenAdj.Scale(nn.config.learningRate, bHiddenAdj)
		bHidden.Add(bHidden, bHiddenAdj)
	}

	return nil
}

func sumAlongAxis(axis int, m *mat.Dense) (*mat.Dense, error) {

	numRows, numCols := m.Dims()

	var output *mat.Dense

	switch axis {
	case 0:
		data := make([]float64, numCols)
		for i := 0; i < numCols; i++ {
			col := mat.Col(nil, i, m)
			data[i] = floats.Sum(col)
		}
		output = mat.NewDense(1, numCols, data)
	case 1:
		data := make([]float64, numRows)
		for i := 0; i < numRows; i++ {
			row := mat.Row(nil, i, m)
			data[i] = floats.Sum(row)
		}
		output = mat.NewDense(numRows, 1, data)
	default:
		return nil, errors.New("invalid axis, must be 0 or 1")
	}

	return output, nil
}

// train обучает нейронную сеть, используя обратное распространение.
func (nn *neuralNet) train(x, y *mat.Dense) error {

	// Инициализируем смещения/веса.
	randSource := rand.NewSource(time.Now().UnixNano())
	randGen := rand.New(randSource)

	wHidden := mat.NewDense(nn.config.inputNeurons, nn.config.hiddenNeurons, nil)
	bHidden := mat.NewDense(1, nn.config.hiddenNeurons, nil)
	wOut := mat.NewDense(nn.config.hiddenNeurons, nn.config.outputNeurons, nil)
	bOut := mat.NewDense(1, nn.config.outputNeurons, nil)

	wHiddenRaw := wHidden.RawMatrix().Data
	bHiddenRaw := bHidden.RawMatrix().Data
	wOutRaw := wOut.RawMatrix().Data
	bOutRaw := bOut.RawMatrix().Data

	for _, param := range [][]float64{
		wHiddenRaw,
		bHiddenRaw,
		wOutRaw,
		bOutRaw,
	} {
		for i := range param {
			param[i] = randGen.Float64()
		}
	}

	// Определяем выход сети.
	output := new(mat.Dense)

	// Используем обратное распространение для регулировки весов и смещений.
	if err := nn.backpropagate(x, y, wHidden, bHidden, wOut, bOut, output); err != nil {
		return err
	}

	// Определяем обученную сеть.
	nn.wHidden = wHidden
	nn.bHidden = bHidden
	nn.wOut = wOut
	nn.bOut = bOut

	return nil
}

// predict делает предсказание с помощью
// обученной нейронной сети.
func (nn *neuralNet) predict(x *mat.Dense) (*mat.Dense, error) {

	// Проверяем, представляет ли значение neuralNet
	// обученную модель.
	if nn.wHidden == nil || nn.wOut == nil {
		return nil, errors.New("the supplied weights are empty")
	}
	if nn.bHidden == nil || nn.bOut == nil {
		return nil, errors.New("the supplied biases are empty")
	}

	// Определяем выход сети.
	output := new(mat.Dense)

	// Завершаем процесс прямого распространения.
	hiddenLayerInput := new(mat.Dense)
	hiddenLayerInput.Mul(x, nn.wHidden)
	addBHidden := func(_, col int, v float64) float64 { return v + nn.bHidden.At(0, col) }
	hiddenLayerInput.Apply(addBHidden, hiddenLayerInput)

	hiddenLayerActivations := new(mat.Dense)
	applySigmoid := func(_, _ int, v float64) float64 { return sigmoid(v) }
	hiddenLayerActivations.Apply(applySigmoid, hiddenLayerInput)

	outputLayerInput := new(mat.Dense)
	outputLayerInput.Mul(hiddenLayerActivations, nn.wOut)
	addBOut := func(_, col int, v float64) float64 { return v + nn.bOut.At(0, col) }
	outputLayerInput.Apply(addBOut, outputLayerInput)
	output.Apply(applySigmoid, outputLayerInput)

	return output, nil
}

func main() {

	// Открываем файл с обучающими данными.
	f, err := os.Open("data/train.csv")
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()

	// Создаём новый CSV-парсер для открытого файла.
	reader := csv.NewReader(f)
	reader.FieldsPerRecord = 7

	// Считываем все записи.
	rawCSVData, err := reader.ReadAll()
	if err != nil {
		log.Fatal(err)
	}

	// inputsData и labelsData будут содержать все
	// числа, которые будут использованы
	// для формирования наших матриц.
	inputsData := make([]float64, 4*len(rawCSVData))
	labelsData := make([]float64, 3*len(rawCSVData))

	// inputsIndex отслеживает текущий индекс
	// значений входных матриц.
	var inputsIndex int
	var labelsIndex int

	// Последовательно помещаем строки в inputsData.
	for idx, record := range rawCSVData {

		// Пропускаем строку заголовков.
		if idx == 0 {
			continue
		}

		// Проходимся в цикле по столбцам.
		for i, val := range record {

			// Преобразуем значение в float.
			parsedVal, err := strconv.ParseFloat(val, 64)
			if err != nil {
				log.Fatal(err)
			}

			// Добавляем значение в labelsData при необходимости.
			if i == 4 || i == 5 || i == 6 {
				labelsData[labelsIndex] = parsedVal
				labelsIndex++
				continue
			}

			// Добавляем значение в inputsData.
			inputsData[inputsIndex] = parsedVal
			inputsIndex++
		}
	}

	// Формируем матрицы.
	inputs := mat.NewDense(len(rawCSVData), 4, inputsData)
	labels := mat.NewDense(len(rawCSVData), 3, labelsData)

	// Определяем архитектуру нашей сети
	// и параметры обучения.
	config := neuralNetConfig{
		inputNeurons:  4,
		outputNeurons: 3,
		hiddenNeurons: 3,
		numEpochs:     5000,
		learningRate:  0.3,
	}

	// Обучаем сеть.
	network := newNetwork(config)
	if err := network.train(inputs, labels); err != nil {
		log.Fatal(err)
	}

	// Form the testing matrices.
	testInputs, testLabels := makeInputsAndLabels("data/test.csv")

	// Делаем предсказание с помощью обученной модели.
	predictions, err := network.predict(testInputs)
	if err != nil {
		log.Fatal(err)
	}

	// Рассчитываем точность модели.
	var truePosNeg int
	numPreds, _ := predictions.Dims()
	for i := 0; i < numPreds; i++ {

		// Получаем вид.
		labelRow := mat.Row(nil, i, testLabels)
		var species int
		for idx, label := range labelRow {
			if label == 1.0 {
				species = idx
				break
			}
		}

		// Считаем количество верных предсказаний.
		if predictions.At(i, species) == floats.Max(mat.Row(nil, i, predictions)) {
			truePosNeg++
		}
	}

	// Подсчитываем точность предсказаний.
	accuracy := float64(truePosNeg) / float64(numPreds)

	// Выводим точность.
	fmt.Printf("\nAccuracy = %0.2f\n\n", accuracy)

}

func makeInputsAndLabels(fileName string) (*mat.Dense, *mat.Dense) {
	// Open the dataset file.
	f, err := os.Open(fileName)
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()

	// Create a new CSV reader reading from the opened file.
	reader := csv.NewReader(f)
	reader.FieldsPerRecord = 7

	// Read in all of the CSV records
	rawCSVData, err := reader.ReadAll()
	if err != nil {
		log.Fatal(err)
	}

	// inputsData and labelsData will hold all the
	// float values that will eventually be
	// used to form matrices.
	inputsData := make([]float64, 4*len(rawCSVData))
	labelsData := make([]float64, 3*len(rawCSVData))

	// Will track the current index of matrix values.
	var inputsIndex int
	var labelsIndex int

	// Sequentially move the rows into a slice of floats.
	for idx, record := range rawCSVData {

		// Skip the header row.
		if idx == 0 {
			continue
		}

		// Loop over the float columns.
		for i, val := range record {

			// Convert the value to a float.
			parsedVal, err := strconv.ParseFloat(val, 64)
			if err != nil {
				log.Fatal(err)
			}

			// Add to the labelsData if relevant.
			if i == 4 || i == 5 || i == 6 {
				labelsData[labelsIndex] = parsedVal
				labelsIndex++
				continue
			}

			// Add the float value to the slice of floats.
			inputsData[inputsIndex] = parsedVal
			inputsIndex++
		}
	}
	inputs := mat.NewDense(len(rawCSVData), 4, inputsData)
	labels := mat.NewDense(len(rawCSVData), 3, labelsData)
	return inputs, labels
}