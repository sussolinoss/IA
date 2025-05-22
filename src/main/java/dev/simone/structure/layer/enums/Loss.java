package dev.simone.structure.layer.enums;


public enum Loss {

    MSE {
        @Override
        public double calculate(double[] predicted, double[] actual) {
            double loss = 0.0;
            for (int i = 0; i < predicted.length; i++) {
                loss += Math.pow(predicted[i] - actual[i], 2);
            }
            return loss / predicted.length;
        }
    },
    BINARY_CROSS_ENTROPY {
        @Override
        public double calculate(double[] predictions, double[] expected) {
            double loss = 0.0;
            for (int i = 0; i < predictions.length; i++) {
                final double y = expected[i];
                final double p = Math.max(1e-15, Math.min(predictions[i], 1 - 1e-15));
                loss += -(y * Math.log(p) + (1 - y) * Math.log(1 - p));
            }
            return loss / predictions.length;
        }
    },
    CROSS_ENTROPY {
        @Override
        public double calculate(double[] predicted, double[] actual) {
            double loss = 0.0;
            for (int i = 0; i < predicted.length; i++) {
                double epsilon = 1e-15;
                predicted[i] = Math.max(Math.min(predicted[i], 1 - epsilon), epsilon);
                loss -= actual[i] * Math.log(predicted[i]) + (1 - actual[i]) * Math.log(1 - predicted[i]);
            }
            return loss / predicted.length;
        }
    };

    public abstract double calculate(double[] predicted, double[] actual);
}