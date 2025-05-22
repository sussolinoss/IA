package dev.simone.structure.layer;

import dev.simone.structure.neuron.Neuron;

public class Layer {

    private final Neuron[] neurons;

    public Layer(final Neuron[] neurons) {
        this.neurons = neurons;
    }

    public double[] evaluate(final double[] inputs) {
        double[] outputs = new double[this.neurons.length];
        for (int i = 0; i < this.neurons.length; i++) {
            outputs[i] = this.neurons[i].process(inputs);
        }
        return outputs;
    }

    public Neuron[] getNeurons() {
        return neurons;
    }
}
