package dev.simone;

import dev.simone.structure.Model;
import dev.simone.structure.layer.Layer;
import dev.simone.structure.layer.enums.Loss;
import dev.simone.structure.neuron.Neuron;
import dev.simone.structure.neuron.enums.Activation;
import dev.simone.structure.neuron.enums.Initializer;

import java.util.Arrays;

public class Main {

    public static void main(String[] args) throws Exception {
        // XOR: input 2 -> hidden 4 -> output 1
        final Model model = new Model(
                Loss.BINARY_CROSS_ENTROPY,
                new Layer(neuron(2, 4, Activation.RELU)),
                new Layer(neuron(4, 1, Activation.SIGMOID))
        );

        // XOR | input
        final double[][] training_inputs = {
                {0, 0},
                {0, 1},
                {1, 0},
                {1, 1}
        };

        // XOR | output
        final double[][] expected_outputs = {
                {0},
                {1},
                {1},
                {0}
        };

        // Train
        model.train(training_inputs, expected_outputs, 10_000, 0.1);

        // Results
        System.out.println();
        System.out.println();
        System.out.println("===========  Training  ===========");
        for (double[] input : training_inputs) {
            double[] output = model.predict(input);
            System.out.printf("| Input: [%d, %d] => Output: %.4f%n",
                    (int) input[0], (int) input[1], output[0]);
        }
        System.out.println("=================================");

        // Space
        System.out.println();

        final double[][] test_inputs = new double[][]{
                {1, 0}
        };
        final double[][] test_output = new double[][]{
                {1}
        };


        System.out.println("===========   TEST   ===========");
        System.out.println("| Input: " + getPrintableValues(test_inputs));
        System.out.println("| Expected: " + getPrintableValues(test_output));
        System.out.printf("| Result => %s", model.evaluate(test_inputs, test_output) + "%");
        System.out.println();
        System.out.println("=================================");
    }

    private static String getPrintableValues(final double[][] values) {
       return Arrays.deepToString(values).replace("[", "").replace("]", "");
    }

    private static Neuron[] neuron(int input_size, int neurons_size, Activation activation) {
        final Neuron[] neurons = new Neuron[neurons_size];
        for (int i = 0; i < neurons_size; i++) {
            neurons[i] = new Neuron(input_size, activation, Initializer.XAVIER);
        }
        return neurons;
    }
}
