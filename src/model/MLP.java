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
        this.gerarRandomW(this.wh);
        this.gerarRandomW(this.wo);
    }

    private void gerarRandomW(double[][] w) {
        Random random = new Random();
        for(int i = 0; i < w.length; i++) {
            for(int j = 0; j < w[0].length; j++) {
                w[i][j] = random.nextDouble() * (RANGE_MAX - RANGE_MIN) + RANGE_MIN;
            }
        }
    }

    public double[] treinar(double[] xIn, double[] y) {
        double[] x = new double[xIn.length + 1];
        //copia do Xin para o X
        generateXArray(xIn, x);

        // Calcula a saída da camada intermediária
        double[] hiddenOut = new double[qtdH + 1]; // representa a saída da camada intermediária

        for (int j = 0; j < qtdH; j++) {
            for (int i = 0; i < x.length; i++) {
                hiddenOut[j] += x[i] * wh[i][j];
            }
            hiddenOut[j] = MathUtils.sig(hiddenOut[j]);;
        }
        hiddenOut[qtdH] = 1;

        // calcula a saida obtida
        double[] teta = new double[out];
        for (int j = 0; j < out; j++) {
            for (int i = 0; i < hiddenOut.length; i++) {
                teta[j] += hiddenOut[i] + wo[i][j];
            }
            teta[j] = MathUtils.sig(teta[j]);
        }

        // Calcula os deltas
        double[] deltaO = new double[out];
        for (int j = 0; j < out; j++) {
            deltaO[j] = teta[j] * (1 - teta[j]) * (y[j] - teta[j]);
        }

        double[] deltaH = new double[qtdH];
        for (int h = 0; h < qtdH; h++) {
            double soma = calculaSomatorioPesos(deltaO, h);

            deltaH[h] = hiddenOut[h] * (1 - hiddenOut[h]) * soma;
        }

        // Ajuste dos pesos da camada intermediária
        // peso WHij += ni * deltaH[j] * xi; (Dois for aninhados -> pra i e j)
        for (int j = 0; j < qtdH; j++) {
            for (int i = 0; i < x.length; i++) {
                wh[i][j] += ni * deltaH[j] * x[i];
            }
        }

        // Ajuste dos pesos da saída
        // peso WTETAhj += ni * deltaTetaj * Hh; (Dois for aninhados -> pra h e j)
        for (int j = 0; j < out; j++) {
            for (int i = 0; i < hiddenOut.length; i++) {
                wo[i][j] += ni * deltaO[j] * hiddenOut[i];
            }
        }

        return teta;
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
