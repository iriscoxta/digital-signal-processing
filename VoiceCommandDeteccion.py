# Aluno(a): Iris Cordeiro Costa 
# Matrícula: 497503
# AP1 - Processamento digital de sinais


# OS GRAFICOS SERÃO PLOTADOS SEGUINDO A ORDEM DAS QUESTÕES 

import scipy.io
from scipy.spatial import distance
import numpy as np
import sounddevice as sd 
import matplotlib.pyplot as plt

# -------------------- CARREGANDO SINAIS (de áudios) P/ TREINAMENTO --------------------
audios_carreg = scipy.io.loadmat('./audios/InputDataTrain.mat')

# -------------------- PEGANDO MATRIZ DE DADOS DOS SINAIS(de áudios) --------------------
audMatriz = audios_carreg['InputDataTrain']

# -------------------- SEPARA AUDIO (NAO) --------------------
 
audio_1 = audMatriz[:, 0]
audio_2 = audMatriz[:, 1]
audio_3 = audMatriz[:, 2]
audio_4 = audMatriz[:, 3]
audio_5 = audMatriz[:, 4]

# -------------------- SEPARA AUDIO (NAO) --------------------
audio_6 = audMatriz[:, 5]
audio_7 = audMatriz[:, 6]
audio_8 = audMatriz[:, 7]
audio_9 = audMatriz[:, 8]
audio_10 = audMatriz[:, 9]


# QUESTAO 01
# ----------------------------------------------------------------------------------
# Carregue os 10 sinais de áudio de InputDataTrain.m e gere os gráficos destes
# sinais, em 2 figuras separadas. Uma figura deve conter os áudios “sim” e a outra
# deve conter os áudios “não”
# ----------------------------------------------------------------------------------

# -------------------- DEFININDO VALORES EIXO X --------------------

x = np.arange(0, audMatriz.shape[0])

# -------------------- CRIANDO FIG P/ GRAFICO AUDIO 'NÃO' --------------------
plt.figure()

# -------------------- PLOT AUDIO 'NÃO' --------------------
plt.plot(x, audio_1, label='audio 01', color='orange', linewidth=0.5)
plt.plot(x, audio_2, label='audio 02', color='darkgreen', linewidth=0.5)
plt.plot(x, audio_3, label='audio 03', color='grey', linewidth=0.5)
plt.plot(x, audio_4, label='audio 04', color='purple', linewidth=0.5)
plt.plot(x, audio_5, label='audio 05', color='mediumblue', linewidth=0.5)

# -------------------- IDENTIFICANDO EIXOS --------------------
plt.xlabel('Tempo')
plt.ylabel('Amplitude')
# -------------------- ADD TITULO GRAFICO --------------------
plt.title('Sinais - áudio "NÃO"')
# -------------------- ADD LEGENDA --------------------
plt.legend()


# -------------------- CRIANDO FIG P/ GRAFICO AUDIO 'SIM' --------------------
plt.figure()

# -------------------- PLOT AUDIO 'SIM' --------------------
plt.plot(x, audio_6, label='audio 06', color='orange', linewidth=0.5)
plt.plot(x, audio_7, label='audio 07', color='darkgreen', linewidth=0.5)
plt.plot(x, audio_8, label='audio 08', color='grey', linewidth=0.5)
plt.plot(x, audio_9, label='audio 09', color='purple', linewidth=0.5)
plt.plot(x, audio_10, label='audio 10', color='mediumblue', linewidth=0.5)

# -------------------- IDENTIFICANDO EIXOS --------------------
plt.xlabel('Tempo')
plt.ylabel('Amplitude')
# -------------------- ADD TITULO GRAFICO --------------------
plt.title('Sinais - áudio "SIM"')
# -------------------- ADD LEGENDA --------------------
plt.legend()

# -------------------- EXIBINDO GRAFICO --------------------
plt.show()



# QUESTAO 02
# ----------------------------------------------------------------------------------
# - Divida cada um destes 10 sinais em 80 blocos de N/80 amostras, em que N é o número de amostras de cada um dos sinais de áudio. 
# - Calcule a energia de cada um destes blocos e gere os gráficos com as energias de destes 80 blocos no eixo y e o
# índice do bloco no eixo x, em 2 figuras separadas. 
# Uma figura deve conter os áudios “sim” e a outra deve conter os áudios “não”. Caso N/80 não seja inteiro, ignore as
# casas decimais de N/P. 
# ----------------------------------------------------------------------------------


# -------------------- DIV. OS SINAIS DE AUDIO (SIM/NAO) EM 80 BLOCOS DE N/80 AMOSTRAS --------------------
divNum = 80
aud1_div = np.array_split(audio_1, divNum)
aud2_div = np.array_split(audio_2, divNum)
aud3_div = np.array_split(audio_3, divNum)
aud4_div = np.array_split(audio_4, divNum)
aud5_div = np.array_split(audio_5, divNum)
aud6_div = np.array_split(audio_6, divNum)
aud7_div = np.array_split(audio_7, divNum)
aud8_div = np.array_split(audio_8, divNum)
aud9_div = np.array_split(audio_9, divNum)
aud10_div = np.array_split(audio_10, divNum)

#  -------------------- INSTANCIANDO VET. P/ ARMAZENAR ENERG. DOS BLOCOS DE SINAIS --------------------
aud1_energ = []
aud2_energ = []
aud3_energ = []
aud4_energ = []
aud5_energ = []
aud6_energ = []
aud7_energ = []
aud8_energ = []
aud9_energ = []
aud10_energ = []

# -------------------- CALCULANDO A ENERG. DE CADA BLOCO (10 SINAIS DE AUDIO) --------------------
for i in range(divNum):
    aud1_energ.append(np.sum(np.square(aud1_div[i])))
    aud2_energ.append(np.sum(np.square(aud2_div[i])))
    aud3_energ.append(np.sum(np.square(aud3_div[i])))
    aud4_energ.append(np.sum(np.square(aud4_div[i])))
    aud5_energ.append(np.sum(np.square(aud5_div[i])))
    aud6_energ.append(np.sum(np.square(aud6_div[i])))
    aud7_energ.append(np.sum(np.square(aud7_div[i])))
    aud8_energ.append(np.sum(np.square(aud8_div[i])))
    aud9_energ.append(np.sum(np.square(aud9_div[i])))
    aud10_energ.append(np.sum(np.square(aud10_div[i])))

# -------------------- DEFININDO VALORES (EIXO X) --------------------
x = np.arange(0, divNum)


# -------------------- CRIANDO FIGG P/ O GRAFICO DE ENERG. AUDIO 'NÃO'--------------------
plt.figure()

# -------------------- PLOT SINAIS DE AUDIO NAO -------------------- 
plt.plot(x, aud1_energ, label='audio 01', color='orange')
plt.plot(x, aud2_energ, label='audio 02', color='darkgreen')
plt.plot(x, aud3_energ, label='audio 03', color='grey')
plt.plot(x, aud4_energ, label='audio 04', color='purple')
plt.plot(x, aud5_energ, label='audio 05', color='mediumblue')

# -------------------- IDENTIFICANDO EIXOS --------------------
plt.xlabel('Bloco')
plt.ylabel('Energia')
# -------------------- ADD TITULO GRAFICO --------------------
plt.title('Energia (sinais de áudio "NÃO")')
# -------------------- ADD LEGENDA --------------------
plt.legend()


# -------------------- CRIANDO FIGG P/ O GRAFICO DE ENERG. AUDIO SIM--------------------
plt.figure()

# -------------------- PLOT SINAIS DE AUDIO SIM -------------------- 
plt.plot(x, aud6_energ, label='audio 06', color='orange')
plt.plot(x, aud7_energ, label='audio 07', color='darkgreen')
plt.plot(x, aud8_energ, label='audio 08', color='grey')
plt.plot(x, aud9_energ, label='audio 09', color='purple')
plt.plot(x, aud10_energ, label='audio 10', color='mediumblue')

# -------------------- IDENTIFICANDO EIXOS --------------------
plt.xlabel('Bloco')
plt.ylabel('Energia')
# -------------------- ADD TITULO GRAFICO --------------------
plt.title('Energia (sinais de áudio "SIM")')
# -------------------- ADD LEGENDA --------------------
plt.legend()
# -------------------- EXIBINDO GRAFICO --------------------
plt.show()


# QUESTAO 03

# ----------------------------------------------------------------------------------
# Calcule o módulo ao quadrado da Transformada de Fourier (TF) dos 10 sinais de
# áudio de InputDataTrain.m e gere os gráficos destas TFs, em 2 figuras separadas.
# Uma figura deve conter os áudios “sim” e a outra deve conter os áudios “não”. O
# eixo x deste gráficos deve corresponder às frequências entre -pi e pi.
# ----------------------------------------------------------------------------------

# -------------------- CALCULANDO MODULO []^2 DA TF DE CADA SINAL DE AUDIO --------------------
aud1_TransF = np.square(np.abs(np.fft.fftshift(np.fft.fft(audio_1))))
aud2_TransF = np.square(np.abs(np.fft.fftshift(np.fft.fft(audio_2))))
aud3_TransF = np.square(np.abs(np.fft.fftshift(np.fft.fft(audio_3))))
aud4_TransF = np.square(np.abs(np.fft.fftshift(np.fft.fft(audio_4))))
aud5_TransF = np.square(np.abs(np.fft.fftshift(np.fft.fft(audio_5))))
aud6_TransF = np.square(np.abs(np.fft.fftshift(np.fft.fft(audio_6))))
aud7_TransF = np.square(np.abs(np.fft.fftshift(np.fft.fft(audio_7))))
aud8_TransF = np.square(np.abs(np.fft.fftshift(np.fft.fft(audio_8))))
aud9_TransF = np.square(np.abs(np.fft.fftshift(np.fft.fft(audio_9))))
aud10_TransF = np.square(np.abs(np.fft.fftshift(np.fft.fft(audio_10))))

# -------------------- DEFININDO VALORES (EIXO X) --------------------
x = np.linspace(-np.pi, np.pi, audMatriz.shape[0])

# -------------------- CRIANDO FIGG P/ O GRAFICO DE TF AUDIO 'NÃO'--------------------
plt.figure()

# -------------------- PLOT TF SINAIS DE AUDIO NAO -------------------- 
plt.plot(x, aud1_TransF, label='audio 01', color='orange')
plt.plot(x, aud2_TransF, label='audio 02', color='darkgreen')
plt.plot(x, aud3_TransF, label='audio 03', color='grey')
plt.plot(x, aud4_TransF, label='audio 04', color='purple')
plt.plot(x, aud5_TransF, label='audio 05', color='mediumblue')

# -------------------- IDENTIFICANDO EIXOS --------------------
plt.xlabel('Frequência')
plt.ylabel('Amplitude')
# -------------------- ADD TITULO GRAFICO --------------------
plt.title('Transformada de Fourier  (sinais de áudio "NÃO")')
# -------------------- ADD LEGENDA --------------------
plt.legend()


# -------------------- CRIANDO FIGG P/ O GRAFICO DE TF AUDIO 'SIM' --------------------
plt.figure()

# -------------------- PLOT TF SINAIS DE AUDIO SIM -------------------- 
plt.plot(x, aud6_TransF, label='audio 06', color='orange')
plt.plot(x, aud7_TransF, label='audio 07', color='darkgreen')
plt.plot(x, aud8_TransF, label='audio 08', color='grey')
plt.plot(x, aud9_TransF, label='audio 09', color='purple')
plt.plot(x, aud10_TransF, label='audio 10', color='mediumblue')

# -------------------- IDENTIFICANDO EIXOS --------------------
plt.xlabel('Frequência')
plt.ylabel('Amplitude')
# -------------------- ADD TITULO GRAFICO --------------------
plt.title('Transformada de Fourier (sinais de áudio "SIM)"')
# -------------------- ADD LEGENDA --------------------
plt.legend()

# -------------------- EXIBINDO GRAFICO --------------------
plt.show()


# QUESTAO 04
# ------------------------------------------------------------------------------------
# Note que o módulo ao quadrado da TF é simétrico em relação à frequência zero.
# Além disso, pode-se perceber que o espectro dos sinais é concentrado nas baixas
# frequências. Desta forma, você deve eliminar as redundâncias das TFs calculadas
# no Item 3 descartando as frequências negativas e as frequências acima de pi/2. Em
# outras palavras, recalcule estas TFs considerando apenas as frequências entre 0 e
# pi/2. Gere os gráficos destas TFs, em 2 figuras separadas. Uma figura deve conter
# os áudios “sim” e a outra deve conter os áudios “não”. O eixo x deste gráficos deve
# corresponder às frequências entre 0 e pi/2.
# ------------------------------------------------------------------------------------

# -------------------- DEF. INDICES DAS FREQ. NO INTERVALO (0 A pi/2) --------------------
xFilt = np.where((x >= 0) & (x <= np.pi/2))[0]

# -------------------- DEF. INTERVALOS DE CORTE DO SINAL (0 a pi/2) --------------------
x_freqCutStart = xFilt[0]
x_freqCutEnd = xFilt[len(xFilt)-1] + 1

# -------------------- FILTRANDO SINAIS TF P/ BAIXAS FREQ. (0 a pi/2) --------------------
aud1_filt_TransF = aud1_TransF[x_freqCutStart:x_freqCutEnd]
aud2_filt_TransF = aud2_TransF[x_freqCutStart:x_freqCutEnd]
aud3_filt_TransF = aud3_TransF[x_freqCutStart:x_freqCutEnd]
aud4_filt_TransF = aud4_TransF[x_freqCutStart:x_freqCutEnd]
aud5_filt_TransF = aud5_TransF[x_freqCutStart:x_freqCutEnd]
aud6_filt_TransF = aud6_TransF[x_freqCutStart:x_freqCutEnd]
aud7_filt_TransF = aud7_TransF[x_freqCutStart:x_freqCutEnd]
aud8_filt_TransF = aud8_TransF[x_freqCutStart:x_freqCutEnd]
aud9_filt_TransF = aud9_TransF[x_freqCutStart:x_freqCutEnd]
aud10_filt_TransF = aud10_TransF[x_freqCutStart:x_freqCutEnd]



# -------------------- CRIANDO FIG P/ GRAFICO DE TF AUDIO NAO (Filtrada) --------------------
plt.figure()

# -------------------- PLOT TF SINAIS DE AUDIO NAO -------------------- 
plt.plot(xFilt, aud1_filt_TransF, label='audio 01', color='orange')
plt.plot(xFilt, aud2_filt_TransF, label='audio 02', color='darkgreen')
plt.plot(xFilt, aud3_filt_TransF, label='audio 03', color='grey')
plt.plot(xFilt, aud3_filt_TransF, label='audio 04', color='purple')
plt.plot(xFilt, aud4_filt_TransF, label='audio 05', color='mediumblue')

# -------------------- IDENTIFICANDO EIXOS --------------------
plt.xlabel('Amostra')
plt.ylabel('Amplitude')
# -------------------- ADD TITULO GRAFICO --------------------
plt.title('Transformada de Fourier (sinais de áudio "NÃO")')
# -------------------- ADD LEGENDA --------------------
plt.legend()

# -------------------- CRIANDO FIG P/ O GRAFICO DE TF AUDIO 'SIM'(Filtrada) --------------------
plt.figure()

# -------------------- PLOT TF SINAIS DE AUDIO SIM -------------------- 
plt.plot(xFilt, aud6_filt_TransF, label='audio 06', color='orange')
plt.plot(xFilt, aud7_filt_TransF, label='audio 07', color='darkgreen')
plt.plot(xFilt, aud8_filt_TransF, label='audio 08', color='grey')
plt.plot(xFilt, aud9_filt_TransF, label='audio 09', color='purple')
plt.plot(xFilt, aud10_filt_TransF, label='audio 10', color='mediumblue')

# -------------------- IDENTIFICANDO EIXOS --------------------
plt.xlabel('Amostra')
plt.ylabel('Amplitude')
# -------------------- ADD TITULO GRAFICO --------------------
plt.title('Transformada de Fourier (sinais de áudio "SIM")')
# -------------------- ADD LEGENDA --------------------
plt.legend()

# -------------------- EXIBINDO GRAFICO --------------------
plt.show()

# QUESTAO 05
# ------------------------------------------------------------------------------------
# Divida cada uma das 10 TFs do Item 4 em 80 blocos de N/320 amostras (N/4 é o
# número de amostras de cada uma das TFs). 
# Calcule a energia de cada um destes blocos 
#  e gere os gráficos com as energias de destes 80 blocos no eixo y e o índice
# do bloco no eixo x, em 2 figuras separadas. Uma figura deve conter os áudios “sim”
# e a outra deve conter os áudios “não”. Caso N/320 não seja inteiro, ignore as casas
# decimais de N/320.
# ------------------------------------------------------------------------------------

# -------------------- DIVIDINDO OS SINAIS DA TRANSFORMADA DE FOURIER DOS AUDIOS (SIM/NAO) EM 80 BLOCOS DE N/320 AMOSTRAS  --------------------
divNum = 80
aud1_div_Transf = np.array_split(aud1_filt_TransF, divNum)
aud2_div_Transf = np.array_split(aud2_filt_TransF, divNum)
aud3_div_Transf = np.array_split(aud3_filt_TransF, divNum)
aud4_div_Transf = np.array_split(aud4_filt_TransF, divNum)
aud5_div_Transf = np.array_split(aud5_filt_TransF, divNum)
aud6_div_Transf = np.array_split(aud6_filt_TransF, divNum)
aud7_div_Transf = np.array_split(aud7_filt_TransF, divNum)
aud8_div_Transf = np.array_split(aud8_filt_TransF, divNum)
aud9_div_Transf = np.array_split(aud9_filt_TransF, divNum)
aud10_div_Transf = np.array_split(aud10_filt_TransF, divNum)

# -------------------- INSTANCIANDO VET. P/ ARMAZENAS ENERG. DOS BLOCOS DE SINAIS --------------------
aud1_Energ_Filt= []
aud2_Energ_Filt = []
aud3_Energ_Filt = []
aud4_Energ_Filt = []
aud5_Energ_Filt = []
aud6_Energ_Filt = []
aud7_Energ_Filt = []
aud8_Energ_Filt = []
aud9_Energ_Filt = []
aud10_Energ_Filt = []

# -------------------- CALCULANDO ENERG DE CADA BLOCO NOS 10 SINAIS DE AUDIO --------------------
for i in range(divNum):
    aud1_Energ_Filt.append(np.sum(aud1_div_Transf[i]))
    aud2_Energ_Filt.append(np.sum(aud2_div_Transf[i]))
    aud3_Energ_Filt.append(np.sum(aud3_div_Transf[i]))
    aud4_Energ_Filt.append(np.sum(aud4_div_Transf[i]))
    aud5_Energ_Filt.append(np.sum(aud5_div_Transf[i]))
    aud6_Energ_Filt.append(np.sum(aud6_div_Transf[i]))
    aud7_Energ_Filt.append(np.sum(aud7_div_Transf[i]))
    aud8_Energ_Filt.append(np.sum(aud8_div_Transf[i]))
    aud9_Energ_Filt.append(np.sum(aud9_div_Transf[i]))
    aud10_Energ_Filt.append(np.sum(aud10_div_Transf[i]))
    
# -------------------- DEF. VALORES EIXO X --------------------
x = np.arange(0, divNum)


# -------------------- CRIANDO FIG P/ O GRAFICO DE ENERG. TF AUDIO NAO --------------------
plt.figure()

# -------------------- PLOT SINAIS DE AUDIO NAO --------------------  
plt.plot(x, aud1_Energ_Filt, label='audio 01', color='orange')
plt.plot(x, aud2_Energ_Filt, label='audio 02', color='darkgreen')
plt.plot(x, aud3_Energ_Filt, label='audio 03', color='grey')
plt.plot(x, aud4_Energ_Filt, label='audio 04', color='purple')
plt.plot(x, aud5_Energ_Filt, label='audio 05', color='mediumblue')

# -------------------- IDENTIFICANDO EIXOS --------------------
plt.xlabel('Bloco')
plt.ylabel('Energia')
# -------------------- ADD TITULO GRAFICO --------------------
plt.title('Energia da Transformada de Fourier (áudios "NÃO)"')
# -------------------- ADD LEGENDA --------------------
plt.legend()


# -------------------- CRIANDO FIG P/ O GRAFICO DE ENERG. TF AUDIO SIM --------------------
plt.figure()

# -------------------- PLOT SINAIS DE AUDIO SIM -------------------- 
plt.plot(x, aud6_Energ_Filt, label='audio 06', color='orange')
plt.plot(x, aud7_Energ_Filt, label='audio 07', color='darkgreen')
plt.plot(x, aud8_Energ_Filt, label='audio 08', color='grey')
plt.plot(x, aud9_Energ_Filt, label='audio 09', color='purple')
plt.plot(x, aud10_Energ_Filt, label='audio 10', color='mediumblue')

# -------------------- IDENTIFICANDO EIXOS --------------------
plt.xlabel('Bloco')
plt.ylabel('Energia')
# -------------------- ADD TITULO GRAFICO --------------------
plt.title('Energia da Transformada de Fourier (áudios "SIM")')
# -------------------- ADD LEGENDA --------------------
plt.legend()

# -------------------- EXIBINDO GRAFICO --------------------
plt.show()

# QUESTAO 06
# -----------------------------------------------------------------------------------------
# - Agora, divida cada um dos sinais de áudio (no domínio do tempo) em 10 blocos
# de N/10 amostras 
# - Calcule o módulo ao quadrado da TF de cada um destes blocos.

# Os 10 espectros resultantes de cada áudio correspondem à Transformada de
# Fourier de tempo curto (short-time Fourier transform – STFT). Tal como no Item 4,
# recalcule estas STFTs considerando apenas as frequências entre 0 e pi/2. Gere os
# gráficos destas STFTs, em 2 figuras separadas, mas apenas para um sinal do tipo
# “sim” e um sinal do tipo “não”. O eixo x deste gráficos deve corresponder às
# frequências entre 0 e pi/2. Note que, no Itens 3 e 4, você deve calcular a TF do sinal
# inteiro (com todas as amostras), enquanto nos Itens 5 e 6 você deve 

# - Calcular a TF dos sub-sinais gerados (cada um com N/10 amostras).
# -----------------------------------------------------------------------------------------


# -------------------- DIVIDINDO CADA SINAL DE AUDIO (SIM/NÃO) em 10 blocos de N/10 amostras --------------------
divNum = 10
aud1_div = np.array_split(audio_1, divNum)
aud2_div = np.array_split(audio_2, divNum)
aud3_div = np.array_split(audio_3, divNum)
aud4_div = np.array_split(audio_4, divNum)
aud5_div = np.array_split(audio_5, divNum)
aud6_div = np.array_split(audio_6, divNum)
aud7_div = np.array_split(audio_7, divNum)
aud8_div = np.array_split(audio_8, divNum)
aud9_div = np.array_split(audio_9, divNum)
aud10_div = np.array_split(audio_10, divNum)


# -------------------- CALCULANDO O MODULO [^2] da TF DE CADA BLOCO DOS SINAIS DE AUDIO --------------------

# -------------------- SHORT-TIME FOURIER TRANSFORM (STFT) --------------------
aud1_STFT = np.square(np.abs(np.fft.fftshift(np.fft.fft(aud1_div))))
aud2_STFT = np.square(np.abs(np.fft.fftshift(np.fft.fft(aud2_div))))
aud3_STFT = np.square(np.abs(np.fft.fftshift(np.fft.fft(aud3_div))))
aud4_STFT = np.square(np.abs(np.fft.fftshift(np.fft.fft(aud4_div))))
aud5_STFT = np.square(np.abs(np.fft.fftshift(np.fft.fft(aud5_div))))
aud6_STFT = np.square(np.abs(np.fft.fftshift(np.fft.fft(aud6_div))))
aud7_STFT = np.square(np.abs(np.fft.fftshift(np.fft.fft(aud7_div))))
aud8_STFT = np.square(np.abs(np.fft.fftshift(np.fft.fft(aud8_div))))
aud9_STFT = np.square(np.abs(np.fft.fftshift(np.fft.fft(aud9_div))))
aud10_STFT = np.square(np.abs(np.fft.fftshift(np.fft.fft(aud10_div))))


# -------------------- DEFININDO VALORES (EIXO X) --------------------
x = np.linspace(-np.pi, np.pi, int(audMatriz.shape[0]/divNum))

# -------------------- DEFININDO INDICES DAS FREQUENCIAS (0 a pi/2) -------------------- 
xFilt = np.where((x >= 0) & (x <= np.pi/2))[0]

# -------------------- DEFININDO INDICES DOS BLOCOS (STFT) --------------------
N_blocs = np.arange(aud1_STFT.shape[0])

# -------------------- FILTRANDO SINAIS P/ BAIXA FREQUENCIA (STFT - 0 A pi/2) -------------------- 
aud1_filT_STFT = aud1_STFT[N_blocs[:, np.newaxis], xFilt]
aud2_filT_STFT = aud2_STFT[N_blocs[:, np.newaxis], xFilt]
aud3_filT_STFT = aud3_STFT[N_blocs[:, np.newaxis], xFilt]
aud4_filT_STFT = aud4_STFT[N_blocs[:, np.newaxis], xFilt]
aud5_filT_STFT = aud5_STFT[N_blocs[:, np.newaxis], xFilt]
aud6_filT_STFT = aud6_STFT[N_blocs[:, np.newaxis], xFilt]
aud7_filT_STFT = aud7_STFT[N_blocs[:, np.newaxis], xFilt]
aud8_filT_STFT = aud8_STFT[N_blocs[:, np.newaxis], xFilt]
aud9_filT_STFT = aud9_STFT[N_blocs[:, np.newaxis], xFilt]
aud10_filT_STFT = aud10_STFT[N_blocs[:, np.newaxis], xFilt]

# -------------------- CRIANDO FIG P/ GRAFICO DA TF FILTRADA (audio NAO) --------------------
plt.figure()

# -------------------- CORES LINHAS GRAFICOS STFT --------------------
lineColors = ['orange', 'darkgreen', 'mediumblue', 'purple', 'gray', 'lime', 'cyan', 'saddlebrown', 'red', 'pink']

# -------------------- PLOT FFT AUDIO NAO --------------------
for i in range(divNum): 
    color = lineColors[i % len(lineColors)]
    plt.plot(xFilt, aud1_filT_STFT[i], label=f'bloco {i+1}', color=color)

# -------------------- IDENTIFICANDO EIXOS --------------------
plt.xlabel('Amostra')
plt.ylabel('Amplitude')
# -------------------- ADD TITULO GRAFICO --------------------
plt.title('Transformada de Fourier de tempo curto (sinal de áudio "NÃO")')
# -------------------- ADD LEGENDA --------------------
plt.legend()

# -------------------- CRIANDO FIG P/ GRAFICO DA TF FILTRADA (audio SIM) --------------------
plt.figure()

# -------------------- PLOT FFT AUDIO SIM --------------------
for i in range(divNum): 
    color = lineColors[i % len(lineColors)]
    plt.plot(xFilt, aud6_filT_STFT[i], label=f'bloco {i+1}', color=color)    

# -------------------- IDENTIFICANDO EIXOS --------------------
plt.xlabel('Amostra')
plt.ylabel('Amplitude')
# -------------------- ADD TITULO GRAFICO --------------------
plt.title('Transformada de Fourier de tempo curto (sinal de áudio "SIM")')
# -------------------- ADD LEGENDA --------------------
plt.legend()

# -------------------- EXIBINDO GRAFICO --------------------
plt.show()

# QUESTAO 07

# Divida estas STFTs em 8 blocos de N/320 amostras (N/40 é o número de
# amostras de uma das STFTs). Calcule a energia de cada um destes blocos. Pra
# cada sinal de áudio, você deve obter 8 energias para cada uma das 10 STFTs,
# totalizando 80 energias para cada sinal de áudio. Não precisa gerar nenhum gráfico
# neste Item 7.
# Note que, nos itens 2, 5 e 7, foram calculadas as energias de 80 blocos em 3
# diferentes domínios. No Item 2, foi usado o domínio do tempo, no Item 5 foi usada a
# TF e no Item 7 a STFT. Estas 80 energias serão usadas para caracterizar o sinal de
# áudio (em cada um dos 3 domínios). O objetivo desta prática é saber, para o caso
# específico do reconhecimento dos comandos de voz “sim” e “não”, qual destes 3
# domínios (tempo, TF e STFT) é o mais eficiente.


# -------------------- DIVIDINDO STFT DOS AUDIOS (SIM/NÃO) EM 8 BLOCOS DE N/320 AMOSTRAS --------------------
divNum = 8

# -------------------- INSTANCIANDO VET P/ ARMAZEAR AS STFT DE CADA BLOCO/8 --------------------  
Div1_Blocos_STFT = []
Div2_Blocos_STFT = []
Div3_Blocos_STFT = []
Div4_Blocos_STFT = []
Div5_Blocos_STFT = []
Div6_Blocos_STFT = []
Div7_Blocos_STFT = []
Div8_Blocos_STFT = []
Div9_Blocos_STFT = []
Div10_Blocos_STFT = []

# -------------------- ARMAZENANDO CADA BLOCO DE STFT/8 --------------------
for i in range(10):
    Div1_Blocos_STFT.append(np.array_split(aud1_STFT[i], divNum))
    Div2_Blocos_STFT.append(np.array_split(aud2_STFT[i], divNum))
    Div3_Blocos_STFT.append(np.array_split(aud3_STFT[i], divNum))
    Div4_Blocos_STFT.append(np.array_split(aud4_STFT[i], divNum))
    Div5_Blocos_STFT.append(np.array_split(aud5_STFT[i], divNum))
    Div6_Blocos_STFT.append(np.array_split(aud6_STFT[i], divNum))
    Div7_Blocos_STFT.append(np.array_split(aud7_STFT[i], divNum))
    Div8_Blocos_STFT.append(np.array_split(aud8_STFT[i], divNum))
    Div9_Blocos_STFT.append(np.array_split(aud9_STFT[i], divNum))
    Div10_Blocos_STFT.append(np.array_split(aud10_STFT[i], divNum))


# INSTANCIANDO VET P/ ARMAZENAR ENERGIAS DE CADA BLOCO (N/320 AMOSTRAS) --------------------


# -------------------- ENERGIAS: 8 ENERG. P/ CADA 1 DAS 10 PARTES DOS STFTs --------------------
Energ1_Blocos_STFT = []
Energ2_Blocos_STFT = []
Energ3_Blocos_STFT = []
Energ4_Blocos_STFT = []
Energ5_Blocos_STFT = []
Energ6_Blocos_STFT = []
Energ7_Blocos_STFT = []
Energ8_Blocos_STFT = []
Energ9_Blocos_STFT = []
Energ10_Blocos_STFT = []

# -------------------- CALCULANDO 80 ENERGIAS (8 ENERG. P/ CADA 1 DAS 10 PARTES DOS STFT) --------------------
for i in range(10):
    for j in range(8):
        Energ1_Blocos_STFT.append(np.sum(np.square(Div1_Blocos_STFT[i][j])))
        Energ2_Blocos_STFT.append(np.sum(np.square(Div2_Blocos_STFT[i][j])))
        Energ3_Blocos_STFT.append(np.sum(np.square(Div3_Blocos_STFT[i][j])))
        Energ4_Blocos_STFT.append(np.sum(np.square(Div4_Blocos_STFT[i][j])))
        Energ5_Blocos_STFT.append(np.sum(np.square(Div5_Blocos_STFT[i][j])))
        Energ6_Blocos_STFT.append(np.sum(np.square(Div6_Blocos_STFT[i][j])))
        Energ7_Blocos_STFT.append(np.sum(np.square(Div7_Blocos_STFT[i][j])))
        Energ8_Blocos_STFT.append(np.sum(np.square(Div8_Blocos_STFT[i][j])))
        Energ9_Blocos_STFT.append(np.sum(np.square(Div9_Blocos_STFT[i][j])))
        Energ10_Blocos_STFT.append(np.sum(np.square(Div10_Blocos_STFT[i][j])))



# QUESTAO 08

# Para cada um dos 3 domínios, organize as 80 energias calculadas em um vetor
# e tire a média destes vetores para as 5 amostas da classe “sim” e para as 5
# amostras da classe “não”. No final, você deverá obter 2 vetores de tamanho 80x1
# (representandos o “sim” e o “não”) para cada um dos 3 domínios. Estes vetores são
# chamados de centroides. Não precisa gerar nenhum gráfico neste Item 8.


# ---------------------------------------- CENTROIDES: ----------------------------------------
    
# -------------------- CALCULANDO MEDIA ENERGIAS DO AUDIO SIM P/ DOM. TEMPO --------------------
medEnerg_N = np.mean(np.array([aud1_energ, aud2_energ, aud3_energ, aud4_energ, aud5_energ]), axis=0)

# -------------------- CALCULANDO MED. ENERGIAS DO AUDIO SIM P/ DOM. TEMPO --------------------
medEnerg_S = np.mean(np.array([aud6_energ, aud7_energ, aud8_energ, aud9_energ, aud10_energ]), axis=0)

# -------------------- CALCULANDO MED. ENERGIAS DO AUDIO NAO P/ DOM. TF --------------------
medEnerg_N_TF = np.mean(np.array([aud1_Energ_Filt, aud2_Energ_Filt, aud3_Energ_Filt, aud4_Energ_Filt, aud5_Energ_Filt]), axis=0)

# -------------------- CALCULANDO MED. ENERGIAS DO AUDIO SIM P/ DOM. TF --------------------
medEnerg_S_TF = np.mean(np.array([aud6_Energ_Filt, aud7_Energ_Filt, aud8_Energ_Filt, aud9_Energ_Filt, aud10_Energ_Filt]), axis=0)

# -------------------- CALCULANDO MED. ENERGIAS DO AUDIO NAO P/ DOM. STFT --------------------
medEnerg_N_SFTF = np.mean(np.array([Energ1_Blocos_STFT, Energ2_Blocos_STFT, Energ3_Blocos_STFT, Energ4_Blocos_STFT, Energ5_Blocos_STFT]), axis=0)

# -------------------- CALCULANDO MED. ENERGIAS DO AUDIO SIM P/ DOM. STFT --------------------
medEnerg_S_SFTF = np.mean(np.array([Energ6_Blocos_STFT, Energ7_Blocos_STFT, Energ8_Blocos_STFT, Energ9_Blocos_STFT, Energ10_Blocos_STFT]), axis=0)


# QUESTAO 09

# Repita os procedimentos do Itens 1 a 7 para calcular os vetores de tamanho
# 80x1 com as energias, só que desta vez usando os 7 sinais de teste do arquivo
# InputDataTest.m e sem gerar nenhum gráfico. Ou seja, apenas gere, para um dos 3
# domínios, os 7 vetores de tamanho 80x1 com energias dos blocos. Não precisa
# gerar nenhum gráfico neste Item 9.


# -------------------- CARREGANDO AUDIOS (TESTE) --------------------
audios_Carreg_Teste = scipy.io.loadmat('./audios/InputDataTest.mat')

# -------------------- PEGANDO MATRIZ DE DADOS DOS SINAIS(de áudios) --------------------
audios_Matriz_Teste = audios_Carreg_Teste['InputDataTest']

# -------------------- SEPARANDO AUDIO (NAO) --------------------
aud1_Test_N = audios_Matriz_Teste[:, 0]
aud2_Test_N = audios_Matriz_Teste[:, 1]
aud3_Test_N = audios_Matriz_Teste[:, 2]

# -------------------- SEPARANDO AUDIO (SIM) --------------------
aud4_Test_S = audios_Matriz_Teste[:, 3]
aud5_Test_S = audios_Matriz_Teste[:, 4]
aud6_Test_S = audios_Matriz_Teste[:, 5]
aud7_Test_S = audios_Matriz_Teste[:, 6]


#------------------------ CANCULANDO ENERGIAS (domínio do tempo) ----------------------

# -------------------- DIVINDO SINAIS DE TESTE (SIM/NAO) EM 80 BLOCOS DE N/80 AMOSTRAS --------------------
divNum = 80
aud1_div_teste = np.array_split(aud1_Test_N, divNum)
aud2_div_teste = np.array_split(aud2_Test_N, divNum)
aud3_div_teste = np.array_split(aud3_Test_N, divNum)
aud4_div_teste = np.array_split(aud4_Test_S, divNum)
aud5_div_teste = np.array_split(aud5_Test_S, divNum)
aud6_div_teste = np.array_split(aud6_Test_S, divNum)
aud7_div_teste = np.array_split(aud7_Test_S, divNum)


# -------------------- INSTANCIANDO VET P/ ARMAZENAR ENERGIAS DOS BLOCOS DE SINAIS --------------------
aud1_Energies_teste = []
aud2_Energies_teste = []
aud3_Energies_teste = []
aud4_Energies_teste = []
aud5_Energies_teste = []
aud6_Energies_teste = []
aud7_Energies_teste = []

# -------------------- CALCULANDO A ENERGIA DE CADA BLOCO NOS 10 SINAIS AUDIO --------------------
for i in range(divNum):
    aud1_Energies_teste.append(np.sum(np.square(aud1_div_teste[i])))
    aud2_Energies_teste.append(np.sum(np.square(aud2_div_teste[i])))
    aud3_Energies_teste.append(np.sum(np.square(aud3_div_teste[i])))
    aud4_Energies_teste.append(np.sum(np.square(aud4_div_teste[i])))
    aud5_Energies_teste.append(np.sum(np.square(aud5_div_teste[i])))
    aud6_Energies_teste.append(np.sum(np.square(aud6_div_teste[i])))
    aud7_Energies_teste.append(np.sum(np.square(aud7_div_teste[i])))



# --------------------------------- CALCULANDO ENERGIAS (dom. transormada de Fourier) ---------------------------------
     
# -------------------- CALCULANDO O MODULO []^2 DA TF DE CADA SINAL (TESTE) --------------------
aud1_TransF = np.square(np.abs(np.fft.fftshift(np.fft.fft(aud1_Test_N))))
aud2_TransF = np.square(np.abs(np.fft.fftshift(np.fft.fft(aud2_Test_N))))
aud3_TransF = np.square(np.abs(np.fft.fftshift(np.fft.fft(aud3_Test_N))))
aud4_TransF = np.square(np.abs(np.fft.fftshift(np.fft.fft(aud4_Test_S))))
aud5_TransF = np.square(np.abs(np.fft.fftshift(np.fft.fft(aud5_Test_S))))
aud6_TransF = np.square(np.abs(np.fft.fftshift(np.fft.fft(aud6_Test_S))))
aud7_TransF = np.square(np.abs(np.fft.fftshift(np.fft.fft(aud7_Test_S))))

# -------------------- DEFININDO VALORES (EIXO X) --------------------
x = np.linspace(-np.pi, np.pi, audios_Matriz_Teste.shape[0])

# -------------------- DEF. INDICES DAS FREQ. NO INTERVALO (0 A pi/2) --------------------
xFilt = np.where((x >= 0) & (x <= np.pi/2))[0]

# -------------------- DEF. INTERVALOS DE CORTE DO SINAL (0 a pi/2) --------------------
x_freqCutStart = xFilt[0]
x_freqCutEnd = xFilt[len(xFilt)-1] + 1

# -------------------- FILTRANDOS OS SINAIS TF P/ BAIXAS FREQ. (0 a pi/2)  --------------------
aud1_TransFourier_filt_test = aud1_TransF[x_freqCutStart:x_freqCutEnd]
aud2_TransFourier_filt_test = aud2_TransF[x_freqCutStart:x_freqCutEnd]
aud3_TransFourier_filt_test = aud3_TransF[x_freqCutStart:x_freqCutEnd]
aud4_TransFourier_filt_test = aud4_TransF[x_freqCutStart:x_freqCutEnd]
aud5_TransFourier_filt_test = aud5_TransF[x_freqCutStart:x_freqCutEnd]
aud6_TransFourier_filt_test = aud6_TransF[x_freqCutStart:x_freqCutEnd]
aud7_TransFourier_filt_test = aud7_TransF[x_freqCutStart:x_freqCutEnd]

# -------------------- DIV. SINAIS TF DOS AUDIOS DE TESTE (SIM/NAO) EM 80 BLOCOS --------------------
divNum = 80
aud1_TransFourier_Teste_Dv = np.array_split(aud1_TransFourier_filt_test, divNum)
aud2_TransFourier_Teste_Dv = np.array_split(aud2_TransFourier_filt_test, divNum)
aud3_TransFourier_Teste_Dv = np.array_split(aud3_TransFourier_filt_test, divNum)
aud4_TransFourier_Teste_Dv = np.array_split(aud4_TransFourier_filt_test, divNum)
aud5_TransFourier_Teste_Dv = np.array_split(aud5_TransFourier_filt_test, divNum)
aud6_TransFourier_Teste_Dv = np.array_split(aud6_TransFourier_filt_test, divNum)
aud7_TransFourier_Teste_Dv = np.array_split(aud7_TransFourier_filt_test, divNum)

# -------------------- INSTANCIANDO VET P/ ARMAZENAR ENERGIAS DOS BLOCOS DE SINAIS --------------------
aud1_TransFourier_TFil_Energ  = []
aud2_TransFourier_TFil_Energ  = []
aud3_TransFourier_TFil_Energ  = []
aud4_TransFourier_TFil_Energ  = []
aud5_TransFourier_TFil_Energ  = []
aud6_TransFourier_TFil_Energ  = []
aud7_TransFourier_TFil_Energ  = []


# -------------------- CALCULANDO A ENERGIA DE CADA BLOCO NOS 10 SINAIS DE AUDIO --------------------
for i in range(divNum):
    aud1_TransFourier_TFil_Energ.append(np.sum(aud1_TransFourier_Teste_Dv[i]))
    aud2_TransFourier_TFil_Energ.append(np.sum(aud2_TransFourier_Teste_Dv[i]))
    aud3_TransFourier_TFil_Energ.append(np.sum(aud3_TransFourier_Teste_Dv[i]))
    aud4_TransFourier_TFil_Energ.append(np.sum(aud4_TransFourier_Teste_Dv[i]))
    aud5_TransFourier_TFil_Energ.append(np.sum(aud5_TransFourier_Teste_Dv[i]))
    aud6_TransFourier_TFil_Energ.append(np.sum(aud6_TransFourier_Teste_Dv[i]))
    aud7_TransFourier_TFil_Energ.append(np.sum(aud7_TransFourier_Teste_Dv[i]))
    
    
# --------------------------------- CALCULA ENERGIA (dom. STFT) ---------------------------------
    
# -------------------- DIVIDINDO OS SINAIS DE TESTE (SIM/NAO) EM 10 BLOCOS DE N/10 AMOSTRAS --------------------
divNum = 10
aud1_div_teste = np.array_split(aud1_Test_N, divNum)
aud2_div_teste = np.array_split(aud2_Test_N, divNum)
aud3_div_teste = np.array_split(aud3_Test_N, divNum)
aud4_div_teste = np.array_split(aud4_Test_S, divNum)
aud5_div_teste = np.array_split(aud5_Test_S, divNum)
aud6_div_teste = np.array_split(aud6_Test_S, divNum)
aud7_div_teste = np.array_split(aud7_Test_S, divNum)


# --------------------------------- CALCULANDO MODULO ^[2] TF (cada bloco dos sinais de teste) ---------------------------------

# -------------------- TRANSFORMADA DE FOURIER DE TEMPO CURTO (short-time Fourier transform – STFT) --------------------
aud1_STFT_teste = np.square(np.abs(np.fft.fftshift(np.fft.fft(aud1_div_teste))))
aud2_STFT_teste = np.square(np.abs(np.fft.fftshift(np.fft.fft(aud2_div_teste))))
aud3_STFT_teste = np.square(np.abs(np.fft.fftshift(np.fft.fft(aud3_div_teste))))
aud4_STFT_teste = np.square(np.abs(np.fft.fftshift(np.fft.fft(aud4_div_teste))))
aud5_STFT_teste = np.square(np.abs(np.fft.fftshift(np.fft.fft(aud5_div_teste))))
aud6_STFT_teste = np.square(np.abs(np.fft.fftshift(np.fft.fft(aud6_div_teste))))
aud7_STFT_teste = np.square(np.abs(np.fft.fftshift(np.fft.fft(aud7_div_teste))))


# -------------------- DEFININDO VALORES (EIXO X) --------------------
x = np.linspace(-np.pi, np.pi, int(audios_Matriz_Teste.shape[0]/divNum))

# -------------------- DEF. INDICES DAS FREQ. NO INTERVALO (0 A pi/2) --------------------
xFilt = np.where((x >= 0) & (x <= np.pi/2))[0]

# -------------------- DEF. INDICES DOS BLOCOS (STFT) --------------------
N_blocs = np.arange(aud1_STFT_teste.shape[0])

# -------------------- FILTRANDO SINAIS STFT P/ BAIXAS FREQ. (0 a pi/2) --------------------
aud1_filT_STFT = aud1_STFT_teste[N_blocs[:, np.newaxis], xFilt]
aud2_filT_STFT = aud2_STFT_teste[N_blocs[:, np.newaxis], xFilt]
aud3_filT_STFT = aud3_STFT_teste[N_blocs[:, np.newaxis], xFilt]
aud4_filT_STFT = aud4_STFT_teste[N_blocs[:, np.newaxis], xFilt]
aud5_filT_STFT = aud5_STFT_teste[N_blocs[:, np.newaxis], xFilt]
aud6_filT_STFT = aud6_STFT_teste[N_blocs[:, np.newaxis], xFilt]
aud7_filT_STFT = aud7_STFT_teste[N_blocs[:, np.newaxis], xFilt]

# -------------------- DIV. as STFT DOS SINAIS DE TESTE (SIM/NÃO) EM 8 BLOCOS DE N/320 AMOSTRAS --------------------
divNum = 8

# -------------------- INSTANCIANDO VET P/ ARMAZENAR CADA BLOCO STFT DIV. POR 8  --------------------
Bloco1_Div_STFT_teste = []
Bloco2_Div_STFT_teste = []
Bloco3_Div_STFT_teste = []
Bloco4_Div_STFT_teste = []
Bloco5_Div_STFT_teste = []
Bloco6_Div_STFT_teste = []
Bloco7_Div_STFT_teste = []

# -------------------- ARMAZENANDO CADA BLOCO DA STFT DIV. POR 8 (10x8) --------------------
for i in range(10):
    Bloco1_Div_STFT_teste.append(np.array_split(aud1_filT_STFT, divNum))
    Bloco2_Div_STFT_teste.append(np.array_split(aud2_filT_STFT, divNum))
    Bloco3_Div_STFT_teste.append(np.array_split(aud3_filT_STFT, divNum))
    Bloco4_Div_STFT_teste.append(np.array_split(aud4_filT_STFT, divNum))
    Bloco5_Div_STFT_teste.append(np.array_split(aud5_filT_STFT, divNum))
    Bloco6_Div_STFT_teste.append(np.array_split(aud6_filT_STFT, divNum))
    Bloco7_Div_STFT_teste.append(np.array_split(aud7_filT_STFT, divNum))


# -------------------- INSTANCIANDO VET P/ ARMAZENAR ENERG. DE CADA BLOCO (N/320 amostras) --------------------


# -------------------- ENERGGIAS: 8 ENERGIAS (p/ cada 1 das 10 STFTs) --------------------
Energ1_Blocos_STFT_Teste = []
Energ2_Blocos_STFT_Teste = []
Energ3_Blocos_STFT_Teste = []
Energ4_Blocos_STFT_Teste = []
Energ5_Blocos_STFT_Teste = []
Energ6_Blocos_STFT_Teste = []
Energ7_Blocos_STFT_Teste = []


# -------------------- CALCULANDO 80 ENERGIAS (8 energias para cada uma das 10 partes dos STFT) --------------------
for i in range(10):
    for j in range(8):
        Energ1_Blocos_STFT_Teste.append(np.sum(np.square(Bloco1_Div_STFT_teste[i][j])))
        Energ2_Blocos_STFT_Teste.append(np.sum(np.square(Bloco2_Div_STFT_teste[i][j])))
        Energ3_Blocos_STFT_Teste.append(np.sum(np.square(Bloco3_Div_STFT_teste[i][j])))
        Energ4_Blocos_STFT_Teste.append(np.sum(np.square(Bloco4_Div_STFT_teste[i][j])))
        Energ5_Blocos_STFT_Teste.append(np.sum(np.square(Bloco5_Div_STFT_teste[i][j])))
        Energ6_Blocos_STFT_Teste.append(np.sum(np.square(Bloco6_Div_STFT_teste[i][j])))
        Energ7_Blocos_STFT_Teste.append(np.sum(np.square(Bloco7_Div_STFT_teste[i][j])))


# QUESTAO 10

# Para realizar a detecção dos comandos de voz (classificação), usaremos o
# chamado Algoritmo do Centroide, que compara os vetores de energia dos áudios de
# teste gerados no Item 9 com os centroides que caracterizam cada classe, gerados
# no Item 8. Desta forma, calcule a distância Euclidiana entre cada um dos 7 vetores
# gerados no Item 9 e os 2 centroides gerados no Item 8, totalizando 14 distâncias
# Euclidianas. 
# Para cada um dos 7 vetores de teste, realize uma escolha entre as classes “sim” e
# “não” baseado no parágrafo anterior e calcule a quantidade de acertos usando cada
# um dos 3 domínios (tempo, TF e STFT). Lembre-se que o arquivo InputDataTest.m
# contém 7 arquivos de áudio, sendo os 3 primeiros correspondentes à palavra “não”
# e os 4 últimos à palavra “sim”.

# -------------------- CALCULO DISTANCIA EUCLIDEANA (dom. Tempo): --------------------

# -------------------- CALCULANDO DIST EUCLIDEANA P/ CENTROIDE NAO (dom. Tempo) --------------------
DistEucl_aud1_DomTemp_Teste_N = distance.euclidean(aud1_Energies_teste,medEnerg_N)
DistEucl_aud2_DomTemp_Teste_N = distance.euclidean(aud2_Energies_teste,medEnerg_N)
DistEucl_aud3_DomTemp_Teste_N = distance.euclidean(aud3_Energies_teste,medEnerg_N)
DistEucl_aud4_DomTemp_Teste_N = distance.euclidean(aud4_Energies_teste,medEnerg_N)
DistEucl_aud5_DomTemp_Teste_N = distance.euclidean(aud5_Energies_teste,medEnerg_N)
DistEucl_aud6_DomTemp_Teste_N = distance.euclidean(aud6_Energies_teste,medEnerg_N)
DistEucl_aud7_DomTemp_Teste_N = distance.euclidean(aud7_Energies_teste,medEnerg_N)
DistEucl_DomTemp_Teste_N = [DistEucl_aud1_DomTemp_Teste_N, DistEucl_aud2_DomTemp_Teste_N, DistEucl_aud3_DomTemp_Teste_N, DistEucl_aud4_DomTemp_Teste_N,
                            DistEucl_aud5_DomTemp_Teste_N, DistEucl_aud6_DomTemp_Teste_N, DistEucl_aud7_DomTemp_Teste_N]

# -------------------- CALCULANDO DIST EUCLIDEANA P/ CENTROIDE SIM  (dom. Tempo) --------------------
DistEucl_aud1_DomTemp_Teste_S = distance.euclidean(aud1_Energies_teste,medEnerg_S)
DistEucl_aud2_DomTemp_Teste_S = distance.euclidean(aud2_Energies_teste,medEnerg_S)
DistEucl_aud3_DomTemp_Teste_S = distance.euclidean(aud3_Energies_teste,medEnerg_S)
DistEucl_aud4_DomTemp_Teste_S = distance.euclidean(aud4_Energies_teste,medEnerg_S)
DistEucl_aud5_DomTemp_Teste_S = distance.euclidean(aud5_Energies_teste,medEnerg_S)
DistEucl_aud6_DomTemp_Teste_S = distance.euclidean(aud6_Energies_teste,medEnerg_S)
DistEucl_aud7_DomTemp_Teste_S = distance.euclidean(aud7_Energies_teste,medEnerg_S)
DistEucl_DomTemp_Teste_S = [DistEucl_aud1_DomTemp_Teste_S, DistEucl_aud2_DomTemp_Teste_S, DistEucl_aud3_DomTemp_Teste_S, DistEucl_aud4_DomTemp_Teste_S,
                            DistEucl_aud5_DomTemp_Teste_S, DistEucl_aud6_DomTemp_Teste_S, DistEucl_aud7_DomTemp_Teste_S]




# -------------------- CALCULO DISTANCIA EUCLIDEANA (dom. TF): --------------------

# -------------------- CALCULANDO DIST EUCLIDEANA P/ CENTROIDE NAO ( dom. TF) --------------------
DistEucl_aud1_TransF_N = distance.euclidean(aud1_TransFourier_TFil_Energ,medEnerg_N_TF)
DistEucl_aud2_TransF_N = distance.euclidean(aud2_TransFourier_TFil_Energ,medEnerg_N_TF)
DistEucl_aud3_TransF_N = distance.euclidean(aud3_TransFourier_TFil_Energ,medEnerg_N_TF)
DistEucl_aud4_TransF_N = distance.euclidean(aud4_TransFourier_TFil_Energ,medEnerg_N_TF)
DistEucl_aud5_TransF_N = distance.euclidean(aud5_TransFourier_TFil_Energ,medEnerg_N_TF)
DistEucl_aud6_TransF_N = distance.euclidean(aud6_TransFourier_TFil_Energ,medEnerg_N_TF)
DistEucl_aud7_TransF_N = distance.euclidean(aud7_TransFourier_TFil_Energ,medEnerg_N_TF)
DistEucl_TransF_N = [DistEucl_aud1_TransF_N, DistEucl_aud2_TransF_N, DistEucl_aud3_TransF_N, DistEucl_aud4_TransF_N,
                     DistEucl_aud5_TransF_N, DistEucl_aud6_TransF_N, DistEucl_aud7_TransF_N]


# -------------------- CALCULANDO DIST EUCLIDEANA P/ CENTROIDE SIM (dom. TF) --------------------
DistEucl_aud1_TransF_S = distance.euclidean(aud1_TransFourier_TFil_Energ,medEnerg_S_TF)
DistEucl_aud2_TransF_S = distance.euclidean(aud2_TransFourier_TFil_Energ,medEnerg_S_TF)
DistEucl_aud3_TransF_S = distance.euclidean(aud3_TransFourier_TFil_Energ,medEnerg_S_TF)
DistEucl_aud4_TransF_S = distance.euclidean(aud4_TransFourier_TFil_Energ,medEnerg_S_TF)
DistEucl_aud5_TransF_S = distance.euclidean(aud5_TransFourier_TFil_Energ,medEnerg_S_TF)
DistEucl_aud6_TransF_S = distance.euclidean(aud6_TransFourier_TFil_Energ,medEnerg_S_TF)
DistEucl_aud7_TransF_S = distance.euclidean(aud7_TransFourier_TFil_Energ,medEnerg_S_TF)
DistEucl_TransF_S = [DistEucl_aud1_TransF_S, DistEucl_aud2_TransF_S, DistEucl_aud3_TransF_S, DistEucl_aud4_TransF_S,
                     DistEucl_aud5_TransF_S, DistEucl_aud6_TransF_S, DistEucl_aud7_TransF_S]


# -------------------- CALCULO DISTANCIA EUCLIDEANA (dom. SFTF): --------------------

# ------------------------------------ CALCULANDO DIST EUCLIDEANA P/ CENTROIDE NAO (dom. SFTF) ------------------------------------
DistEucl_aud1_SFTransF_N = distance.euclidean(Energ1_Blocos_STFT_Teste, medEnerg_N_SFTF)
DistEucl_aud2_SFTransF_N = distance.euclidean(Energ2_Blocos_STFT_Teste, medEnerg_N_SFTF)
DistEucl_aud3_SFTransF_N = distance.euclidean(Energ3_Blocos_STFT_Teste, medEnerg_N_SFTF)
DistEucl_aud4_SFTransF_N = distance.euclidean(Energ4_Blocos_STFT_Teste, medEnerg_N_SFTF)
DistEucl_aud5_SFTransF_N = distance.euclidean(Energ5_Blocos_STFT_Teste, medEnerg_N_SFTF)
DistEucl_aud6_SFTransF_N = distance.euclidean(Energ6_Blocos_STFT_Teste, medEnerg_N_SFTF)
DistEucl_aud7_SFTransF_N = distance.euclidean(Energ7_Blocos_STFT_Teste, medEnerg_N_SFTF)
DistEucl_SFTransF_N = [DistEucl_aud1_SFTransF_N, DistEucl_aud2_SFTransF_N, DistEucl_aud3_SFTransF_N, DistEucl_aud4_SFTransF_N,
                       DistEucl_aud5_SFTransF_N, DistEucl_aud6_SFTransF_N, DistEucl_aud7_SFTransF_N]


DistEucl_aud1_SFTransF_S = distance.euclidean(Energ1_Blocos_STFT_Teste, medEnerg_S_SFTF)
DistEucl_aud2_SFTransF_S = distance.euclidean(Energ2_Blocos_STFT_Teste, medEnerg_S_SFTF)
DistEucl_aud3_SFTransF_S = distance.euclidean(Energ3_Blocos_STFT_Teste, medEnerg_S_SFTF)
DistEucl_aud4_SFTransF_S = distance.euclidean(Energ4_Blocos_STFT_Teste, medEnerg_S_SFTF)
DistEucl_aud5_SFTransF_S = distance.euclidean(Energ5_Blocos_STFT_Teste, medEnerg_S_SFTF)
DistEucl_aud6_SFTransF_S = distance.euclidean(Energ6_Blocos_STFT_Teste, medEnerg_S_SFTF)
DistEucl_aud7_SFTransF_S = distance.euclidean(Energ7_Blocos_STFT_Teste, medEnerg_S_SFTF)
DistEucl_SFTransF_S = [DistEucl_aud1_SFTransF_S, DistEucl_aud2_SFTransF_S, DistEucl_aud3_SFTransF_S, DistEucl_aud4_SFTransF_S,
                       DistEucl_aud5_SFTransF_S, DistEucl_aud6_SFTransF_S, DistEucl_aud7_SFTransF_S]


Dominio_Tempo = 0 
Dominio_TransF = 0
Dominio_STTransF = 0

# ----------------------- COMPARANDO AUDIOS NÃO (ACERTOS) --------------------
for i in range(3):
    if(DistEucl_DomTemp_Teste_N[i]<DistEucl_DomTemp_Teste_S[i]):
        Dominio_Tempo += 1      
    if(DistEucl_TransF_N[i]<DistEucl_TransF_S[i]):
        Dominio_TransF += 1 
    if(DistEucl_SFTransF_N[i]<DistEucl_SFTransF_S[i]):
        Dominio_STTransF += 1 

# ----------------------- COMPARANDO AUDIOS NÃO (ACERTOS) --------------------
for i in range(3,7):
    if(DistEucl_DomTemp_Teste_N[i]>DistEucl_DomTemp_Teste_S[i]):
        Dominio_Tempo += 1
    if(DistEucl_TransF_N[i]>DistEucl_TransF_S[i]):
        Dominio_TransF += 1 
    if(DistEucl_SFTransF_N[i]>DistEucl_SFTransF_S[i]):
        Dominio_STTransF += 1 


print('Acertos do dominio do tempo:' , Dominio_Tempo)
print('Acertos do dominio da Tf:', Dominio_TransF)
print('Acertos do dominio da STFT:', Dominio_STTransF)


# Qual domínio obteve um maior número de acertos? Qual obteve o menor número de acertos? 

# O domínio que obteve maior número de acertos foi: domínio da Transformada de Fourier.

# Analisando os resultados entre Transformada de Fourier e Tempo (Dom. TF x Dom. Tempo):
# Para reconhecer os áudios da palavra "SIM" e da palavra "NÃO" é necessário:
# - analisar em que frequências as amostras dos sinais são predomiantes.
# O domínio do tempo não realiza essa analise

# Analisando os resultados entre Transformada de Fourier e Transforma de Fourier de Tempo Curto  (Dom. TF x Dom. STFT):
# A STFT já obteve resultado melhor que o domínio do tempo porém não tão bom quanto da  TF. 
# Essa é usada principalmente para analisar e representar sinais que variam no tempo e na frequência, como sinais de áudio e de comunicação.
# Ela é uma extensão da Transformada de Fourier clássica, que é usada para representar sinais no domínio da frequência, 
# mas não considera as variações temporais ao longo do sinal.
# A STFT leva em conta como as características do áudio se alteram com o tempo. (não é uma boa esolha p/ fazer analise nesse caso)
# 
# Já a Transformada analise cada frequencia de cada classe, logo obteve o melhor resultado entre as 3 opções