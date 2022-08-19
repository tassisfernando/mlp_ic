package model;

import math.MathUtils;

import java.util.Random;

import static java.lang.System.arraycopy;

public class MLP {
    private int in, out, qtdH;

    private double[][] wh;
    private double[][] wo;
    private double ni;

    private final static double RANGE_MIN = -0.03;
    private final static double RANGE_MAX = 0.03;

    public MLP(int in, int out, int qtdH, double ni) {
        this.in = in;
        this.out = out;
        this.qtdH = qtdH;
        this.ni = ni;

        this.wh = new double[in + 1][qtdH];
        this.wo = new double[qtdH + 1][out];

        //inicializar os pesos: pegar do código anterior
        this.gerarRandomW();
    }

    private void gerarRandomW() {
        Random random = new Random();
        for(int i = 0; i < wh.length; i++) {
            for(int j = 0; j < wh[0].length; j++) {
                wh[i][j] = random.nextDouble() * (RANGE_MAX - RANGE_MIN) + RANGE_MIN;
            }
        }
    }

    public double[] treinar(double[] xIn, double[] y) {
        double[] x = new double[xIn.length + 1];
        //copia do Xin para o X
        generateXArray(xIn, x);

        // Calcula a saída da camada intermediária
        double[] H = new double[qtdH + 1]; // representa a saída da camada intermediária

        for (int j = 0; j < qtdH; j++) {
            for (int i = 0; i < x.length; i++) {
                H[j] += x[i] * wh[i][j];
            }
            H[j] = MathUtils.sig(H[j]);;
        }
        H[qtdH] = 1;

        // calcula a saida obtida
        double[] teta = new double[out];
        for (int j = 0; j < out; j++) {
            for (int i = 0; i < H.length; i++) {
                teta[j] += H[i] + wo[i][j];
            }
            teta[j] = MathUtils.sig(teta[j]);
        }

        // Calcula os deltas
        double[] deltaO = new double[out];
        for (int j = 0; j < out; j++) {
            deltaO[j] = teta[j] * (1 - teta[j]) * (y[j] - teta[j]);
        }

        double[] deltaH = new double[qtdH];
        for(int h = 0; h < qtdH; h++) {
            double soma = calculaSomatorioPesos(deltaO, h);

            deltaH[h] = H[h] * (1 - H[h]) * soma;
        }

        // Ajuste dos pesos da camada intermediária
        // peso WHij += ni * deltaHn * xi; (Dois for aninhados -> pra i e j)
        // peso WTETAhj += ni * deltaTetaj * Hh; (Dois for aninhados -> pra h e j)

        return new double[1];
    }

    private double calculaSomatorioPesos(double[] deltaO, int h) {
        double soma = 0d;
        for(int j = 0; j < out; j++) {
            soma += deltaO[j] * wo[h][j];
        }

        return soma;
    }

    private void generateXArray(double[] xIn, double[] x) {
        arraycopy(xIn, 0, x, 0, xIn.length);
        x[x.length - 1] = 1D;
    }
}
