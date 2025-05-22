package dev.simone.structure;

import dev.simone.structure.layer.Layer;
import dev.simone.structure.layer.enums.Loss;
import dev.simone.structure.neuron.Neuron;

import java.util.ArrayList;
import java.util.List;

public final class Model {

    private final Loss loss;
    private final Layer[] layers;

    public Model(final Loss loss, final Layer... layers) {
        this.loss = loss;
        this.layers = layers;
    }

    public double[] predict(final double[] inputs) {
        double[] current = inputs;
        for (final Layer layer : layers) {
            current = layer.evaluate(current);
        }
        return current;
    }

    public double evaluate(double[][] inputs, double[][] expectedOutputs) {
        int correct = 0;
        int total = inputs.length;

        for (int i = 0; i < total; i++) {
            double[] prediction = this.predict(inputs[i]);
            int predictedClass = prediction[0] >= 0.5 ? 1 : 0;
            int trueClass = (int) expectedOutputs[i][0];

            if (predictedClass == trueClass) {
                correct++;
            }
        }
        return 100.0 * correct / total;
    }


    public void train(final double[][] training_inputs, final double[][] expected_outputs, final int epochs, final double learning_rate) throws Exception {
        for (int epoch = 0; epoch < epochs; epoch++) {
            double loss_tot = 0.0;
            int correct = 0;

            final long now = System.currentTimeMillis();

            for (int i = 0; i < training_inputs.length; i++) {
                final List<double[]> layer_inputs = new ArrayList<>();
                final List<double[]> layer_outputs = new ArrayList<>();

                double[] current = training_inputs[i];
                layer_inputs.add(current);

                for (final Layer layer : layers) {
                    current = layer.evaluate(current);
                    layer_inputs.add(current);
                    layer_outputs.add(current);
                }

                final double[] prediction = current;
                loss_tot += loss.calculate(prediction, expected_outputs[i]);

                if ((prediction[0] >= 0.5 && expected_outputs[i][0] == 1) ||
                    (prediction[0] < 0.5 && expected_outputs[i][0] == 0)) {
                    correct++;
                }

                this.backpropagation(layer_inputs, layer_outputs, expected_outputs[i], learning_rate);
            }

            if (epoch % 50 == 0) {
                final long timestamp = System.currentTimeMillis() - now;
                final double loss =  loss_tot / training_inputs.length;
                final double accuracy = (double) correct / training_inputs.length;

                System.out.printf("\n==== Epoch x%d ====", epoch);
                System.out.printf("\n| Loss: %.6f", loss);
                System.out.printf("\n| Accuracy: %s ", accuracy * 100);
                System.out.printf("\n| Time: %sms", timestamp);
            }
        }
    }

    private void backpropagation(final List<double[]> layer_inputs,
                                 final List<double[]> layer_outputs,
                                 final double[] expected_outputs,
                                 final double learning_rate) throws Exception {

        double[] error = new double[expected_outputs.length];
        final double[] outputs = layer_outputs.get(layer_outputs.size() - 1);

        for (int i = 0; i < error.length; i++) {
            error[i] = outputs[i] - expected_outputs[i];
        }

        if (this.layers.length > layer_inputs.size()) {
            throw new Exception("this.layers.length > layer_inputs.size()");
        }

        for (int i = this.layers.length - 1; i >= 0; i--) {
            final Layer layer = this.layers[i];
            final double[] inputs = layer_inputs.get(i);
            final double[] new_error = new double[inputs.length];
            final Neuron[] neurons = layer.getNeurons();

            for (int j = 0; j < neurons.length; j++) {
                final Neuron neuron = neurons[j];

                double z = 0.0;
                for (int k = 0; k < inputs.length; k++) {
                    z += inputs[k] * neuron.getWeights()[k];
                }
                z += neuron.getBias();

                final double delta = error[j] * neuron.getActivation().derivative(z);

                final double[] weights = neuron.getWeights().clone();

                for (int k = 0; k < inputs.length; k++) {
                    neuron.getWeights()[k] -= learning_rate * delta * inputs[k];
                    new_error[k] += weights[k] * delta;
                }

                neuron.setBias(neuron.getBias() - learning_rate * delta);
            }


            error = new_error;
        }
    }
}