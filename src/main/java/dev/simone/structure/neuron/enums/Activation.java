package dev.simone.structure.neuron.enums;


public enum Activation {
    SIGMOID {
        @Override
        public double calculate(double x) {
            return 1.0 / (1.0 + Math.exp(-x));
        }

        @Override
        public double derivative(double x) {
            double sig = calculate(x);
            return sig * (1 - sig);
        }
    },
    RELU {
        @Override
        public double calculate(double x) {
            return Math.max(0, x);
        }

        @Override
        public double derivative(double x) {
            return x > 0 ? 1 : 0;
        }
    },
    MISH {
        @Override
        public double calculate(double x) {
            double soft = Math.log(1 + Math.exp(x));
            double tanh = Math.tanh(soft);
            return x * tanh;
        }

        @Override
        public double derivative(double x) {
            double sp = Math.log(1 + Math.exp(x));
            double tanh_sp = Math.tanh(sp);
            return x * (1 - tanh_sp * tanh_sp) * (Math.exp(x) / (1 + Math.exp(x))) + tanh_sp;
        }
    },
    LINEAR {
        @Override
        public double calculate(double x) {
            return x;
        }

        @Override
        public double derivative(double x) {
            return 1;
        }
    };

    public abstract double calculate(double x);
    public abstract double derivative(double x);
}