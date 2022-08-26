package app;

import model.MLP;

import static java.lang.Math.abs;

public class MLPRunner {

//    /* AND */
//    private static final double[][][] DATABASE = {
//            { { 0D, 0D }, { 0D } },
//            { { 0D, 1D }, { 0D } },
//            { { 1D, 0D }, { 0D } },
//            { { 1D, 1D }, { 1D } }
//    };


    /* XOR */
    private static final double[][][] DATABASE = {
            { { 0D, 0D }, { 0D } },
            { { 0D, 1D }, { 1D } },
            { { 1D, 0D }, { 1D } },
            { { 1D, 1D }, { 0D } }
    };


    /* OR */
//    private static final double[][][] DATABASE = {
//            { { 0D, 0D }, { 0D } },
//            { { 0D, 1D }, { 1D } },
//            { { 1D, 0D }, { 1D } },
//            { { 1D, 1D }, { 1D } }
//    };


    /* Robô */
//    private static final double[][][] DATABASE = {
//            { { 0D, 0D, 0D }, { 1D, 1D } },
//            { { 0D, 0D, 1D }, { 0D, 1D } },
//            { { 0D, 1D, 0D }, { 1D, 0D } },
//            { { 0D, 1D, 1D }, { 0D, 1D } },
//            { { 1D, 0D, 0D }, { 1D, 0D } },
//            { { 1D, 0D, 1D }, { 1D, 0D } },
//            { { 1D, 1D, 0D }, { 1D, 0D } },
//            { { 1D, 1D, 1D }, { 1D, 0D } }
//    };

    public static void main(String[] args) {
        final double NI = 0.1;
        final int N_EPOCAS = 10000;

        int qtdH = 2;
        MLP mlp = new MLP(DATABASE[0][0].length, DATABASE[0][1].length, qtdH, NI);
        double erroEp, erroAm = 0D;

        for(int e = 0; e < N_EPOCAS; e++) {
            erroEp = 0D;
            for (double[][] sample : DATABASE) {
                double[] x = sample[0];
                double[] y = sample[1];
                double[] out = mlp.treinar(x, y);

                erroAm = sumErro(y, out);
                erroEp += erroAm;
            }

            System.out.printf("Época: %s - Erro: %6f\n", (e+1), erroEp);
        }
    }

    private static double sumErro(double[] y, double[] out) {
        double sum = 0D;
        for(int i = 0; i < y.length; i++) {
            sum += abs(y[i] - out[i]);
        }

        return sum;
    }
}
