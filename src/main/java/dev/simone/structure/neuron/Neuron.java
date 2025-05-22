package dev.simone.structure.neuron;

import dev.simone.structure.neuron.enums.Activation;
import dev.simone.structure.neuron.enums.Initializer;

public class Neuron {

    private double bias;
    private double[] weights;
    private final Activation activation;

    public Neuron(final int inputs_size, final Activation activation, final Initializer initializer) {
        this.bias = 0.0;
        this.activation = activation;
        this.weights = new double[inputs_size];

        for (int i = 0; i < inputs_size; i++) {
            this.weights[i] = initializer.getWeight();
        }
    }

    public double process(final double[] inputs) {
        if (inputs.length != this.weights.length) {
            throw new IllegalArgumentException("[Neuron] Input lengths don't match");
        }

        double sum = 0.0;
        for (int i = 0; i < inputs.length; i++) {
            sum += inputs[i] * this.weights[i];
        }

        sum += this.bias;
        return this.activation.calculate(sum);
    }

    public double[] getWeights() {
        return this.weights;
    }

    public double getBias() {
        return this.bias;
    }

    public Activation getActivation() {
        return this.activation;
    }

    public void setBias(double bias) {
        this.bias = bias;
    }
}
