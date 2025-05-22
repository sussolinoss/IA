package dev.simone.structure.neuron.enums;


import java.util.Random;

public enum Initializer {

    XAVIER {
        @Override
        public double getWeight() {
            return new Random().nextGaussian() * 0.1;
        }
    };

    public abstract double getWeight();
}