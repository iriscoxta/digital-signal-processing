# Aluno(a): Iris Cordeiro Costa 
# Matrícula: 497503
# AP1 - Processamento digital de sinais

import scipy.io
from scipy.spatial import distance
import numpy as np
import sounddevice as sd 
import matplotlib.pyplot as plt

# Carrega os sinais de áudios para o treinamento
audios_carreg = scipy.io.loadmat('./audios/InputDataTrain.mat')

# Pega a matriz de dados dos sinais de áudio
audMatriz = audios_carreg['InputDataTrain']

# Separa os sinais de áudio 'NÃO'
 
audio_1 = audMatriz[:, 0]
audio_2 = audMatriz[:, 1]
audio_3 = audMatriz[:, 2]
audio_4 = audMatriz[:, 3]
audio_5 = audMatriz[:, 4]

# Separa os sinais de áudio 'SIM'
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

# -------------------- Criando figura p/ gráfico de áudios 'NÃO' --------------------
plt.figure()

# -------------------- Plotando os sinais de áudio 'NÃO' --------------------
plt.plot(x, audio_1, label='audio 01', color='orange', linewidth=0.5)
plt.plot(x, audio_2, label='audio 02', color='darkgreen', linewidth=0.5)
plt.plot(x, audio_3, label='audio 03', color='grey', linewidth=0.5)
plt.plot(x, audio_4, label='audio 04', color='purple', linewidth=0.5)
plt.plot(x, audio_5, label='audio 05', color='mediumblue', linewidth=0.5)

# -------------------- Identificando eixos --------------------
plt.xlabel('Tempo')
plt.ylabel('Amplitude')

# -------------------- título do gráfico --------------------
plt.title('Sinais - áudio "NÃO"')
# -------------------- Legenda do gráfico --------------------
plt.legend()


# -------------------- Criando figura p/ gráfico de áudios 'SIM' --------------------
plt.figure()

# -------------------- Plotando os sinais de áudio 'SIM' --------------------
plt.plot(x, audio_6, label='audio 06', color='orange', linewidth=0.5)
plt.plot(x, audio_7, label='audio 07', color='darkgreen', linewidth=0.5)
plt.plot(x, audio_8, label='audio 08', color='grey', linewidth=0.5)
plt.plot(x, audio_9, label='audio 09', color='purple', linewidth=0.5)
plt.plot(x, audio_10, label='audio 10', color='mediumblue', linewidth=0.5)

# -------------------- Identificando eixos --------------------
plt.xlabel('Tempo')
plt.ylabel('Amplitude')
# -------------------- título do gráfico --------------------
plt.title('Sinais - áudio "SIM"')
# -------------------- Legenda do gráfico --------------------
plt.legend()

#  -------------------- Exibindo os gráficos --------------------
plt.show()



# QUESTAO 02
# ----------------------------------------------------------------------------------
# Divida cada um destes 10 sinais em 80 blocos de N/80 amostras, em que N é o
# número de amostras de cada um dos sinais de áudio. Calcule a energia de cada um
# destes blocos e gere os gráficos com as energias de destes 80 blocos no eixo y e o
# índice do bloco no eixo x, em 2 figuras separadas. Uma figura deve conter os áudios
# “sim” e a outra deve conter os áudios “não”. Caso N/80 não seja inteiro, ignore as
# casas decimais de N/P. 
# ----------------------------------------------------------------------------------


# -------------------- Dividir os sinais de áudio 'SIM' e 'NÃO' em 80 blocos de N/80 amostras --------------------
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

#  -------------------- Instânciando vetores p/ armazenar as energias dos blocos de sinais  --------------------
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

# -------------------- Calculando a energia de cada bloco nos 10 sinais de áudio --------------------
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

# -------------------- Definindo valores do eixo X --------------------
x = np.arange(0, divNum)


# -------------------- Criando uma figura para o gráfico de energias do áudios 'NÃO'--------------------
plt.figure()

# -------------------- Plotando os sinais de áudio 'NÃO' --------------------
plt.plot(x, aud1_energ, label='audio 01', color='orange')
plt.plot(x, aud2_energ, label='audio 02', color='darkgreen')
plt.plot(x, aud3_energ, label='audio 03', color='grey')
plt.plot(x, aud4_energ, label='audio 04', color='purple')
plt.plot(x, aud5_energ, label='audio 05', color='mediumblue')

# -------------------- Identificando eixos --------------------
plt.xlabel('Bloco')
plt.ylabel('Energia')
# -------------------- add um título ao gráfico --------------------
plt.title('Energia (sinais de áudio "NÃO")')
# -------------------- add uma legenda --------------------
plt.legend()


# -------------------- Criando uma figura para o gráfico de energias do áudios 'SIM' --------------------
plt.figure()

# -------------------- Plotando os sinais de áudio 'SIM' --------------------
plt.plot(x, aud6_energ, label='audio 06', color='orange')
plt.plot(x, aud7_energ, label='audio 07', color='darkgreen')
plt.plot(x, aud8_energ, label='audio 08', color='grey')
plt.plot(x, aud9_energ, label='audio 09', color='purple')
plt.plot(x, aud10_energ, label='audio 10', color='mediumblue')

# -------------------- Identificando eixos --------------------
plt.xlabel('Bloco')
plt.ylabel('Energia')
# -------------------- add um título ao gráfico --------------------
plt.title('Energia (sinais de áudio "SIM")')
# -------------------- add uma legenda --------------------
plt.legend()
# -------------------- Exibindo os gráficos --------------------
plt.show()


# QUESTAO 03

# ----------------------------------------------------------------------------------
# Calcule o módulo ao quadrado da Transformada de Fourier (TF) dos 10 sinais de
# áudio de InputDataTrain.m e gere os gráficos destas TFs, em 2 figuras separadas.
# Uma figura deve conter os áudios “sim” e a outra deve conter os áudios “não”. O
# eixo x deste gráficos deve corresponder às frequências entre -pi e pi.
# ----------------------------------------------------------------------------------

# -------------------- Calculando o módulo ao quadrado da transformada de Fourier de cada sinal de áudio --------------------
aud1_TransF = np.square(np.fft.fftshift(np.fft.fft(audio_1)))
aud2_TransF = np.square(np.fft.fftshift(np.fft.fft(audio_2)))
aud3_TransF = np.square(np.fft.fftshift(np.fft.fft(audio_3)))
aud4_TransF = np.square(np.fft.fftshift(np.fft.fft(audio_4)))
aud5_TransF = np.square(np.fft.fftshift(np.fft.fft(audio_5)))
aud6_TransF = np.square(np.fft.fftshift(np.fft.fft(audio_6)))
aud7_TransF = np.square(np.fft.fftshift(np.fft.fft(audio_7)))
aud8_TransF = np.square(np.fft.fftshift(np.fft.fft(audio_8)))
aud9_TransF = np.square(np.fft.fftshift(np.fft.fft(audio_9)))
aud10_TransF = np.square(np.fft.fftshift(np.fft.fft(audio_10)))

# -------------------- Definindo valores do eixo X --------------------
x = np.linspace(-np.pi, np.pi, audMatriz.shape[0])

# -------------------- Criando uma figura para o gráfico de transformada de Fourier dos áudios 'NÃO'--------------------
plt.figure()

#  -------------------- Plotando fft dos sinais de áudio 'NÃO' --------------------
plt.plot(x, aud1_TransF, label='audio 01', color='orange')
plt.plot(x, aud2_TransF, label='audio 02', color='darkgreen')
plt.plot(x, aud3_TransF, label='audio 03', color='grey')
plt.plot(x, aud4_TransF, label='audio 04', color='purple')
plt.plot(x, aud5_TransF, label='audio 05', color='mediumblue')

#  -------------------- Identificando eixos--------------------
plt.xlabel('Frequência')
plt.ylabel('Amplitude')
# -------------------- add título ao gráfico --------------------
plt.title('Transformada de Fourier  (sinais de áudio "NÃO")')
# -------------------- Adicionando legenda --------------------
plt.legend()


# -------------------- Criando uma figura para o gráfico de transformada de Fourier dos áudios 'SIM' --------------------
plt.figure()

# --------------------Plotando fft dos sinais de áudio 'SIM' --------------------
plt.plot(x, aud6_TransF, label='audio 06', color='orange')
plt.plot(x, aud7_TransF, label='audio 07', color='darkgreen')
plt.plot(x, aud8_TransF, label='audio 08', color='grey')
plt.plot(x, aud9_TransF, label='audio 09', color='purple')
plt.plot(x, aud10_TransF, label='audio 10', color='mediumblue')

# -------------------- Identificando eixos --------------------
plt.xlabel('Frequência')
plt.ylabel('Amplitude')
# -------------------- add título ao gráfico --------------------
plt.title('Transformada de Fourier (sinais de áudio "SIM)"')
# -------------------- Adicionando legenda --------------------
plt.legend()

# -------------------- Exibindo os gráficos --------------------
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

# -------------------- Definindo os indices das freqências no intervalo de 0 a pi/2 --------------------
xFilt = np.where((x >= 0) & (x <= np.pi/2))[0]

# -------------------- Definindo os intervalos de corte do sinal (0 a pi/2 ) --------------------
x_freqCutStart = xFilt[0]
x_freqCutEnd = xFilt[len(xFilt)-1] + 1

# -------------------- Filtrando os sinais FT para as baixas frequências (0 a pi/2 ) --------------------
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



# -------------------- Criar uma figura para o gráfico de transformada de Fourier dos áudios 'NÃO' (Filtrada) --------------------
plt.figure()

# -------------------- Plotando fft dos sinais de áudio 'NÃO' --------------------
plt.plot(xFilt, aud1_filt_TransF, label='audio 01', color='orange')
plt.plot(xFilt, aud2_filt_TransF, label='audio 02', color='darkgreen')
plt.plot(xFilt, aud3_filt_TransF, label='audio 03', color='grey')
plt.plot(xFilt, aud3_filt_TransF, label='audio 04', color='purple')
plt.plot(xFilt, aud4_filt_TransF, label='audio 05', color='mediumblue')

# -------------------- Identificando eixos --------------------
plt.xlabel('Amostra')
plt.ylabel('Amplitude')
# -------------------- add título ao gráfico --------------------
plt.title('Transformada de Fourier (sinais de áudio "NÃO")')
# -------------------- Adicionando uma legenda --------------------
plt.legend()

# -------------------- Criando uma figura para o gráfico de transformada de Fourier dos áudios 'SIM'(Filtrada) --------------------
plt.figure()

# -------------------- Plotando fft dos sinais de áudio 'SIM' --------------------
plt.plot(xFilt, aud6_filt_TransF, label='audio 06', color='orange')
plt.plot(xFilt, aud7_filt_TransF, label='audio 07', color='darkgreen')
plt.plot(xFilt, aud8_filt_TransF, label='audio 08', color='grey')
plt.plot(xFilt, aud9_filt_TransF, label='audio 09', color='purple')
plt.plot(xFilt, aud10_filt_TransF, label='audio 10', color='mediumblue')

# -------------------- Identificando eixos --------------------
plt.xlabel('Amostra')
plt.ylabel('Amplitude')
# -------------------- add título ao gráfico --------------------
plt.title('Transformada de Fourier (sinais de áudio "SIM")')
# -------------------- Adicionando legenda --------------------
plt.legend()

# -------------------- Exibindo os gráficos --------------------
plt.show()

# QUESTAO 05
# ------------------------------------------------------------------------------------
# Divida cada uma das 10 TFs do Item 4 em 80 blocos de N/320 amostras (N/4 é o
# número de amostras de cada uma das TFs). Calcule a energia de cada um destes
# blocos e gere os gráficos com as energias de destes 80 blocos no eixo y e o índice
# do bloco no eixo x, em 2 figuras separadas. Uma figura deve conter os áudios “sim”
# e a outra deve conter os áudios “não”. Caso N/320 não seja inteiro, ignore as casas
# decimais de N/320.
# ------------------------------------------------------------------------------------

# Dividir os sinais da TF dos áudios 'SIM' e 'NÃO' em 80 blocos de N/320 amostras
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

# Instânciar vetores para armazenar as energias dos blocos de sinais
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

# Calcular a energia de cada bloco nos 10 sinais de áudio
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
    
# Definir valores do eixo X
x = np.arange(0, divNum)


# Criar uma figura para o gráfico de energias da TF dos áudios 'NÃO'
plt.figure()

# Plotar os sinais de áudio 'NÃO' 
plt.plot(x, aud1_Energ_Filt, label='audio 01', color='orange')
plt.plot(x, aud2_Energ_Filt, label='audio 02', color='darkgreen')
plt.plot(x, aud3_Energ_Filt, label='audio 03', color='grey')
plt.plot(x, aud4_Energ_Filt, label='audio 04', color='purple')
plt.plot(x, aud5_Energ_Filt, label='audio 05', color='mediumblue')

# Identificando eixos
plt.xlabel('Bloco')
plt.ylabel('Energia')
# add título ao gráfico
plt.title('Energia da Transformada de Fourier (áudios "NÃO)"')
# Adicionando legenda
plt.legend()


# Criar uma figura para o gráfico de energias da TF dos áudios 'SIM'
plt.figure()

# Plotar os sinais de áudio 'SIM' 
plt.plot(x, aud6_Energ_Filt, label='audio 06', color='orange')
plt.plot(x, aud7_Energ_Filt, label='audio 07', color='darkgreen')
plt.plot(x, aud8_Energ_Filt, label='audio 08', color='grey')
plt.plot(x, aud9_Energ_Filt, label='audio 09', color='purple')
plt.plot(x, aud10_Energ_Filt, label='audio 10', color='mediumblue')

# Identificando eixos
plt.xlabel('Bloco')
plt.ylabel('Energia')
# add título ao gráfico
plt.title('Energia da Transformada de Fourier (áudios "SIM")')
# Adicionando legenda
plt.legend()

# Exibir os gráficos
plt.show()

# QUESTAO 06
# -----------------------------------------------------------------------------------------
# Agora, divida cada um dos sinais de áudio (no domínio do tempo) em 10 blocos
# de N/10 amostras e calcule o módulo ao quadrado da TF de cada um destes blocos.
# Os 10 espectros resultantes de cada áudio correspondem à Transformada de
# Fourier de tempo curto (short-time Fourier transform – STFT). Tal como no Item 4,
# recalcule estas STFTs considerando apenas as frequências entre 0 e pi/2. Gere os
# gráficos destas STFTs, em 2 figuras separadas, mas apenas para um sinal do tipo
# “sim” e um sinal do tipo “não”. O eixo x deste gráficos deve corresponder às
# frequências entre 0 e pi/2. Note que, no Itens 3 e 4, você deve calcular a TF do sinal
# inteiro (com todas as amostras), enquanto nos Itens 5 e 6 você deve calcular a TF
# dos sub-sinais gerados (cada um com N/10 amostras).
# -----------------------------------------------------------------------------------------


# Dividir os sinais de áudio 'SIM' e 'NÃO' em 10 blocos de N/10 amostras
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


# Calcular o módulo ao quadrado da transformada de Fourier de cada bloco dos sinais de áudio
# Transformada de Fourier de tempo curto (short-time Fourier transform – STFT)
audio01_STFT = np.square(np.fft.fftshift(np.fft.fft(aud1_div)))
audio02_STFT = np.square(np.fft.fftshift(np.fft.fft(aud2_div)))
audio03_STFT = np.square(np.fft.fftshift(np.fft.fft(aud3_div)))
audio04_STFT = np.square(np.fft.fftshift(np.fft.fft(aud4_div)))
audio05_STFT = np.square(np.fft.fftshift(np.fft.fft(aud5_div)))
audio06_STFT = np.square(np.fft.fftshift(np.fft.fft(aud6_div)))
audio07_STFT = np.square(np.fft.fftshift(np.fft.fft(aud7_div)))
audio08_STFT = np.square(np.fft.fftshift(np.fft.fft(aud8_div)))
audio09_STFT = np.square(np.fft.fftshift(np.fft.fft(aud9_div)))
audio10_STFT = np.square(np.fft.fftshift(np.fft.fft(aud10_div)))


# Definir valores do eixo X
x = np.linspace(-np.pi, np.pi, int(audMatriz.shape[0]/divNum))

# Definir os indices das freqências no intervalo de 0 a pi/2 
xFilt = np.where((x >= 0) & (x <= np.pi/2))[0]

#Definir os índices dos blocos da STFT
N_blocs = np.arange(audio01_STFT.shape[0])

# Filtrando os sinais da STFT para as baixas frequências (0 a pi/2 ) 
aud1_filT_STFT = audio01_STFT[N_blocs[:, np.newaxis], xFilt]
aud2_filT_STFT = audio02_STFT[N_blocs[:, np.newaxis], xFilt]
aud3_filT_STFT = audio03_STFT[N_blocs[:, np.newaxis], xFilt]
aud4_filT_STFT = audio04_STFT[N_blocs[:, np.newaxis], xFilt]
aud5_filT_STFT = audio05_STFT[N_blocs[:, np.newaxis], xFilt]
aud6_filT_STFT = audio06_STFT[N_blocs[:, np.newaxis], xFilt]
aud7_filT_STFT = audio07_STFT[N_blocs[:, np.newaxis], xFilt]
aud8_filT_STFT = audio08_STFT[N_blocs[:, np.newaxis], xFilt]
aud9_filT_STFT = audio09_STFT[N_blocs[:, np.newaxis], xFilt]
aud10_filT_STFT = audio10_STFT[N_blocs[:, np.newaxis], xFilt]

# Criando uma figura para o gráfico de transformada de Fourier dos áudios 'NÃO' (Filtrada)
plt.figure()

#Definindo cores para as linhas dos gráficos de STFT
lineColors = ['orange', 'darkgreen', 'mediumblue', 'purple', 'gray', 'lime', 'cyan', 'saddlebrown', 'red', 'pink']

# Plotando fft dos sinais de áudio 'NÃO'
for i in range(divNum): 
    color = lineColors[i % len(lineColors)]
    plt.plot(xFilt, aud1_filT_STFT[i], label=f'bloco {i+1}', color=color)

# Identificando eixos
plt.xlabel('Amostra')
plt.ylabel('Amplitude')
# Add título ao gráfico
plt.title('Transformada de Fourier de tempo curto (sinal de áudio "NÃO")')
# Add legenda
plt.legend()

# Criando uma figura para o gráfico de transformada de Fourier dos áudios 'SIM'(Filtrada)
plt.figure()

# Plotando fft dos sinais de áudio 'SIM'
for i in range(divNum): 
    color = lineColors[i % len(lineColors)]
    plt.plot(xFilt, aud6_filT_STFT[i], label=f'bloco {i+1}', color=color)    

# Identificando eixos
plt.xlabel('Amostra')
plt.ylabel('Amplitude')
# add título ao gráfico
plt.title('Transformada de Fourier de tempo curto (sinal de áudio "SIM")')
# Adicionando legenda
plt.legend()

# Exibir os gráficos
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


# Dividir as STFT dos sinais de áudio 'SIM' e 'NÃO' em 8 blocos de N/320 amostras
divNum = 8

# Instânciar vetores para armazenar de cada bloco da STFT dividido por 8  
STFT01DividedBlocs = []
STFT02DividedBlocs = []
STFT03DividedBlocs = []
STFT04DividedBlocs = []
STFT05DividedBlocs = []
STFT06DividedBlocs = []
STFT07DividedBlocs = []
STFT08DividedBlocs = []
STFT09DividedBlocs = []
STFT10DividedBlocs = []

# Armazenar cada bloco da STFT dividido por 8 (10x8x730)
for i in range(10):
    STFT01DividedBlocs.append(np.array_split(audio01_STFT[i], divNum))
    STFT02DividedBlocs.append(np.array_split(audio02_STFT[i], divNum))
    STFT03DividedBlocs.append(np.array_split(audio03_STFT[i], divNum))
    STFT04DividedBlocs.append(np.array_split(audio04_STFT[i], divNum))
    STFT05DividedBlocs.append(np.array_split(audio05_STFT[i], divNum))
    STFT06DividedBlocs.append(np.array_split(audio06_STFT[i], divNum))
    STFT07DividedBlocs.append(np.array_split(audio07_STFT[i], divNum))
    STFT08DividedBlocs.append(np.array_split(audio08_STFT[i], divNum))
    STFT09DividedBlocs.append(np.array_split(audio09_STFT[i], divNum))
    STFT10DividedBlocs.append(np.array_split(audio10_STFT[i], divNum))


# Instânciar vetores para armazenar as energias de cada bloco (N/320 amostras)
# Energias: 8 energias para cada uma das 10 STFTs
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

# Calculando as 80 energias 8 energias para cada uma das 10 partes dos STFT
for i in range(10):
    for j in range(8):
        Energ1_Blocos_STFT.append(np.sum(np.square(STFT01DividedBlocs[i][j])))
        Energ2_Blocos_STFT.append(np.sum(np.square(STFT02DividedBlocs[i][j])))
        Energ3_Blocos_STFT.append(np.sum(np.square(STFT03DividedBlocs[i][j])))
        Energ4_Blocos_STFT.append(np.sum(np.square(STFT04DividedBlocs[i][j])))
        Energ5_Blocos_STFT.append(np.sum(np.square(STFT05DividedBlocs[i][j])))
        Energ6_Blocos_STFT.append(np.sum(np.square(STFT06DividedBlocs[i][j])))
        Energ7_Blocos_STFT.append(np.sum(np.square(STFT07DividedBlocs[i][j])))
        Energ8_Blocos_STFT.append(np.sum(np.square(STFT08DividedBlocs[i][j])))
        Energ9_Blocos_STFT.append(np.sum(np.square(STFT09DividedBlocs[i][j])))
        Energ10_Blocos_STFT.append(np.sum(np.square(STFT10DividedBlocs[i][j])))



# QUESTAO 08

# Para cada um dos 3 domínios, organize as 80 energias calculadas em um vetor
# e tire a média destes vetores para as 5 amostas da classe “sim” e para as 5
# amostras da classe “não”. No final, você deverá obter 2 vetores de tamanho 80x1
# (representandos o “sim” e o “não”) para cada um dos 3 domínios. Estes vetores são
# chamados de centroides. Não precisa gerar nenhum gráfico neste Item 8.

# -------------------- Energias do domínio do tempo --------------------
aud1_energ 
aud2_energ
aud3_energ
aud4_energ
aud5_energ
aud6_energ
aud7_energ
aud8_energ
aud9_energ
aud10_energ

# -------------------- Energias do domínio da TF --------------------
aud1_Energ_Filt
aud2_Energ_Filt
aud3_Energ_Filt
aud4_Energ_Filt
aud5_Energ_Filt
aud6_Energ_Filt
aud7_Energ_Filt
aud8_Energ_Filt
aud9_Energ_Filt
aud10_Energ_Filt

# -------------------- Energias do domínio da STFT --------------------
Energ1_Blocos_STFT
Energ2_Blocos_STFT
Energ3_Blocos_STFT
Energ4_Blocos_STFT
Energ5_Blocos_STFT
Energ6_Blocos_STFT
Energ7_Blocos_STFT
Energ8_Blocos_STFT
Energ9_Blocos_STFT
Energ10_Blocos_STFT


# CENTROIDES:
    
# -------------------- Calculando média das energias do áudio "NAO" para o domínio do tempo --------------------
medEnerg_N = np.mean(np.array([aud1_energ, aud2_energ, aud3_energ, aud4_energ, aud5_energ]), axis=0)

# -------------------- Calculando média das energias do áudio "SIM" para o domínio do tempo --------------------
medEnerg_S = np.mean(np.array([aud6_energ, aud7_energ, aud8_energ, aud9_energ, aud10_energ]), axis=0)

# -------------------- Calculandp média das energias do áudio "NAO" para o domínio de TF --------------------
medEnerg_N_TF = np.mean(np.array([aud1_Energ_Filt, aud2_Energ_Filt, aud3_Energ_Filt, aud4_Energ_Filt, aud5_Energ_Filt]), axis=0)

# -------------------- Calculando média das energias do áudio "SIM" para o domínio de TF --------------------
medEnerg_S_TF = np.mean(np.array([aud6_Energ_Filt, aud7_Energ_Filt, aud8_Energ_Filt, aud9_Energ_Filt, aud10_Energ_Filt]), axis=0)

# -------------------- Calculando média das energias do áudio "NAO" para o domínio de STFT --------------------
medEnerg_N_SFTF = np.mean(np.array([Energ1_Blocos_STFT, Energ2_Blocos_STFT, Energ3_Blocos_STFT, Energ4_Blocos_STFT, Energ5_Blocos_STFT]), axis=0)

# -------------------- Calculando média das energias do áudio "SIM" para o domínio de STFT --------------------
medEnerg_S_SFTF = np.mean(np.array([Energ6_Blocos_STFT, Energ7_Blocos_STFT, Energ8_Blocos_STFT, Energ9_Blocos_STFT, Energ10_Blocos_STFT]), axis=0)


# QUESTAO 09

# Repita os procedimentos do Itens 1 a 7 para calcular os vetores de tamanho
# 80x1 com as energias, só que desta vez usando os 7 sinais de teste do arquivo
# InputDataTest.m e sem gerar nenhum gráfico. Ou seja, apenas gere, para um dos 3
# domínios, os 7 vetores de tamanho 80x1 com energias dos blocos. Não precisa
# gerar nenhum gráfico neste Item 9.


# -------------------- Carregando os sinais de áudios para o teste --------------------
audios_Carreg_Teste = scipy.io.loadmat('./audios/InputDataTest.mat')

# -------------------- Pegando a matriz de dados dos sinais de áudio --------------------
audios_Matriz_Teste = audios_Carreg_Teste['InputDataTest']

# -------------------- Separando sinais de áudio 'NÃO' --------------------
aud1_Test_N = audios_Matriz_Teste[:, 0]
aud2_Test_N = audios_Matriz_Teste[:, 1]
aud3_Test_N = audios_Matriz_Teste[:, 2]

# -------------------- Separando sinais de áudio 'SIM' --------------------
aud4_Test_S = audios_Matriz_Teste[:, 3]
aud5_Test_S = audios_Matriz_Teste[:, 4]
aud6_Test_S = audios_Matriz_Teste[:, 5]
aud7_Test_S = audios_Matriz_Teste[:, 6]


#------------------------ CANCULANDO ENERGIAS (domínio do tempo) ----------------------

# -------------------- Dividindo os sinais de teste 'SIM' e 'NÃO' em 80 blocos de N/80 amostras --------------------
divNum = 80
aud1_div_teste = np.array_split(aud1_Test_N, divNum)
aud2_div_teste = np.array_split(aud2_Test_N, divNum)
aud3_div_teste = np.array_split(aud3_Test_N, divNum)
aud4_div_teste = np.array_split(aud4_Test_S, divNum)
aud5_div_teste = np.array_split(aud5_Test_S, divNum)
aud6_div_teste = np.array_split(aud6_Test_S, divNum)
aud7_div_teste = np.array_split(aud7_Test_S, divNum)


# -------------------- Instância vetores para armazenar as energias dos blocos de sinais --------------------
aud1_Energies_teste = []
aud2_Energies_teste = []
aud3_Energies_teste = []
aud4_Energies_teste = []
aud5_Energies_teste = []
aud6_Energies_teste = []
aud7_Energies_teste = []

# -------------------- Calculando a energia de cada bloco nos 10 sinais de áudio --------------------
for i in range(divNum):
    aud1_Energies_teste.append(np.sum(np.square(aud1_div_teste[i])))
    aud2_Energies_teste.append(np.sum(np.square(aud2_div_teste[i])))
    aud3_Energies_teste.append(np.sum(np.square(aud3_div_teste[i])))
    aud4_Energies_teste.append(np.sum(np.square(aud4_div_teste[i])))
    aud5_Energies_teste.append(np.sum(np.square(aud5_div_teste[i])))
    aud6_Energies_teste.append(np.sum(np.square(aud6_div_teste[i])))
    aud7_Energies_teste.append(np.sum(np.square(aud7_div_teste[i])))



# --------------------------------- CALCULANDO ENERGIAS (dom. transormada de Fourier) ---------------------------------
     
# -------------------- Calculando o módulo ao quadrado da transformada de Fourier de cada sinal de teste --------------------
aud1_TransF = np.abs(np.fft.fftshift(np.fft.fft(aud1_Test_N)))**2
aud2_TransF = np.abs(np.fft.fftshift(np.fft.fft(aud2_Test_N)))**2
aud3_TransF = np.abs(np.fft.fftshift(np.fft.fft(aud3_Test_N)))**2
aud4_TransF = np.abs(np.fft.fftshift(np.fft.fft(aud4_Test_S)))**2
aud5_TransF = np.abs(np.fft.fftshift(np.fft.fft(aud5_Test_S)))**2
aud6_TransF = np.abs(np.fft.fftshift(np.fft.fft(aud6_Test_S)))**2
aud7_TransF = np.abs(np.fft.fftshift(np.fft.fft(aud7_Test_S)))**2

# -------------------- Definindo valores do eixo X --------------------
x = np.linspace(-np.pi, np.pi, audios_Matriz_Teste.shape[0])

# -------------------- Definindo os indices das freqências no intervalo de 0 a pi/2 --------------------
xFilt = np.where((x >= 0) & (x <= np.pi/2))[0]

# -------------------- Definindo os intervalos de corte do sinal (0 a pi/2 ) --------------------
x_freqCutStart = xFilt[0]
x_freqCutEnd = xFilt[len(xFilt)-1] + 1

# -------------------- Filtrando os sinais FT para as baixas frequências (0 a pi/2)  --------------------
aud1_TransFourier_filt_test = aud1_TransF[x_freqCutStart:x_freqCutEnd]
aud2_TransFourier_filt_test = aud2_TransF[x_freqCutStart:x_freqCutEnd]
aud3_TransFourier_filt_test = aud3_TransF[x_freqCutStart:x_freqCutEnd]
aud4_TransFourier_filt_test = aud4_TransF[x_freqCutStart:x_freqCutEnd]
aud5_TransFourier_filt_test = aud5_TransF[x_freqCutStart:x_freqCutEnd]
aud6_TransFourier_filt_test = aud6_TransF[x_freqCutStart:x_freqCutEnd]
aud7_TransFourier_filt_test = aud7_TransF[x_freqCutStart:x_freqCutEnd]

# -------------------- Dividindo os sinais da TF dos áudios de teste 'SIM' e 'NÃO' em 80 blocos --------------------
divNum = 80
aud1_TransFourier_Teste_Dv = np.array_split(aud1_TransFourier_filt_test, divNum)
aud2_TransFourier_Teste_Dv = np.array_split(aud2_TransFourier_filt_test, divNum)
aud3_TransFourier_Teste_Dv = np.array_split(aud3_TransFourier_filt_test, divNum)
aud4_TransFourier_Teste_Dv = np.array_split(aud4_TransFourier_filt_test, divNum)
aud5_TransFourier_Teste_Dv = np.array_split(aud5_TransFourier_filt_test, divNum)
aud6_TransFourier_Teste_Dv = np.array_split(aud6_TransFourier_filt_test, divNum)
aud7_TransFourier_Teste_Dv = np.array_split(aud7_TransFourier_filt_test, divNum)

# -------------------- Instânciando vetores para armazenar as energias dos blocos de sinais --------------------
aud1_TransFourier_TFil_Energ = []
aud2_TransFourier_TFil_Energ  = []
aud3_TransFourier_TFil_Energ  = []
aud4_TransFourier_TFil_Energ  = []
aud5_TransFourier_TFil_Energ  = []
aud6_TransFourier_TFil_Energ  = []
aud7_TransFourier_TFil_Energ  = []


# -------------------- Calculando a energia de cada bloco nos 10 sinais de áudio --------------------
for i in range(divNum):
    aud1_TransFourier_TFil_Energ .append(np.sum(np.square(aud1_TransFourier_Teste_Dv[i])))
    aud2_TransFourier_TFil_Energ .append(np.sum(np.square(aud2_TransFourier_Teste_Dv[i])))
    aud3_TransFourier_TFil_Energ .append(np.sum(np.square(aud3_TransFourier_Teste_Dv[i])))
    aud4_TransFourier_TFil_Energ .append(np.sum(np.square(aud4_TransFourier_Teste_Dv[i])))
    aud5_TransFourier_TFil_Energ .append(np.sum(np.square(aud5_TransFourier_Teste_Dv[i])))
    aud6_TransFourier_TFil_Energ .append(np.sum(np.square(aud6_TransFourier_Teste_Dv[i])))
    aud7_TransFourier_TFil_Energ .append(np.sum(np.square(aud7_TransFourier_Teste_Dv[i])))
    
    
# --------------------------------- CALCULA ENERGIA (dom. STFT) ---------------------------------
    
# -------------------- Dividindo os sinais de teste 'SIM' e 'NÃO' em 10 blocos de N/10 amostras --------------------
divNum = 10
aud1_div_teste = np.array_split(aud1_Test_N, divNum)
aud2_div_teste = np.array_split(aud2_Test_N, divNum)
aud3_div_teste = np.array_split(aud3_Test_N, divNum)
aud4_div_teste = np.array_split(aud4_Test_S, divNum)
aud5_div_teste = np.array_split(aud5_Test_S, divNum)
aud6_div_teste = np.array_split(aud6_Test_S, divNum)
aud7_div_teste = np.array_split(aud7_Test_S, divNum)


# --------------------------------- CALCULANDO MODULO ^[2] TF (cada bloco dos sinais de teste) ---------------------------------

# -------------------- Transformada de Fourier de tempo curto (short-time Fourier transform – STFT) --------------------
audio01Test_STFT = np.abs(np.fft.fftshift(np.fft.fft(aud1_div_teste)))**2
audio02Test_STFT = np.abs(np.fft.fftshift(np.fft.fft(aud2_div_teste)))**2
audio03Test_STFT = np.abs(np.fft.fftshift(np.fft.fft(aud3_div_teste)))**2
audio04Test_STFT = np.abs(np.fft.fftshift(np.fft.fft(aud4_div_teste)))**2
audio05Test_STFT = np.abs(np.fft.fftshift(np.fft.fft(aud5_div_teste)))**2
audio06Test_STFT = np.abs(np.fft.fftshift(np.fft.fft(aud6_div_teste)))**2
audio07Test_STFT = np.abs(np.fft.fftshift(np.fft.fft(aud7_div_teste)))**2


# -------------------- Definindo valores do eixo X --------------------
x = np.linspace(-np.pi, np.pi, int(audios_Matriz_Teste.shape[0]/divNum))

# -------------------- Definindo os indices das freqências no intervalo de 0 a pi/2 --------------------
xFilt = np.where((x >= 0) & (x <= np.pi/2))[0]

# -------------------- Definindo os índices dos blocos da STFT --------------------
N_blocs = np.arange(audio01Test_STFT.shape[0])

# -------------------- Filtrando os sinais da STFT para as baixas frequências (0 a pi/2 ) --------------------
aud1_filT_STFT = audio01Test_STFT[N_blocs[:, np.newaxis], xFilt]
aud2_filT_STFT = audio02Test_STFT[N_blocs[:, np.newaxis], xFilt]
aud3_filT_STFT = audio03Test_STFT[N_blocs[:, np.newaxis], xFilt]
aud4_filT_STFT = audio04Test_STFT[N_blocs[:, np.newaxis], xFilt]
aud5_filT_STFT = audio05Test_STFT[N_blocs[:, np.newaxis], xFilt]
aud6_filT_STFT = audio06Test_STFT[N_blocs[:, np.newaxis], xFilt]
aud7_filT_STFT = audio07Test_STFT[N_blocs[:, np.newaxis], xFilt]

# -------------------- Dividindo as STFT dos sinais de teste 'SIM' e 'NÃO' em 8 blocos de N/320 amostras --------------------
divNum = 8

# -------------------- Instânciando vetores para armazenar de cada bloco da STFT dividido por 8  --------------------
STFT01TestDividedBlocs = []
STFT02TestDividedBlocs = []
STFT03TestDividedBlocs = []
STFT04TestDividedBlocs = []
STFT05TestDividedBlocs = []
STFT06TestDividedBlocs = []
STFT07TestDividedBlocs = []

# -------------------- Armazenando cada bloco da STFT dividido por 8 (10x8) --------------------
for i in range(10):
    STFT01TestDividedBlocs.append(np.array_split(aud1_filT_STFT, divNum))
    STFT02TestDividedBlocs.append(np.array_split(aud2_filT_STFT, divNum))
    STFT03TestDividedBlocs.append(np.array_split(aud3_filT_STFT, divNum))
    STFT04TestDividedBlocs.append(np.array_split(aud4_filT_STFT, divNum))
    STFT05TestDividedBlocs.append(np.array_split(aud5_filT_STFT, divNum))
    STFT06TestDividedBlocs.append(np.array_split(aud6_filT_STFT, divNum))
    STFT07TestDividedBlocs.append(np.array_split(aud7_filT_STFT, divNum))


# -------------------- Instânciando vetores p/ armazenar as energias de cada bloco (N/320 amostras) --------------------
# -------------------- Energias: 8 energias para cada uma das 10 STFTs --------------------
STFT01TestBlocsEnergy = []
STFT02TestBlocsEnergy = []
STFT03TestBlocsEnergy = []
STFT04TestBlocsEnergy = []
STFT05TestBlocsEnergy = []
STFT06TestBlocsEnergy = []
STFT07TestBlocsEnergy = []


# -------------------- Calculando 80 energias (8 energias para cada uma das 10 partes dos STFT) --------------------
for i in range(10):
    for j in range(8):
        STFT01TestBlocsEnergy.append(np.sum(np.square(STFT01TestDividedBlocs[i][j])))
        STFT02TestBlocsEnergy.append(np.sum(np.square(STFT02TestDividedBlocs[i][j])))
        STFT03TestBlocsEnergy.append(np.sum(np.square(STFT03TestDividedBlocs[i][j])))
        STFT04TestBlocsEnergy.append(np.sum(np.square(STFT04TestDividedBlocs[i][j])))
        STFT05TestBlocsEnergy.append(np.sum(np.square(STFT05TestDividedBlocs[i][j])))
        STFT06TestBlocsEnergy.append(np.sum(np.square(STFT06TestDividedBlocs[i][j])))
        STFT07TestBlocsEnergy.append(np.sum(np.square(STFT07TestDividedBlocs[i][j])))


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

# -------------------- Calculando a distância Euclidiana para a centroide NÃO (dom. Tempo) --------------------
DistEucl_aud1_DomTemp_Teste_N = distance.euclidean(aud1_Energies_teste,medEnerg_N)
DistEucl_aud2_DomTemp_Teste_N = distance.euclidean(aud2_Energies_teste,medEnerg_N)
DistEucl_aud3_DomTemp_Teste_N = distance.euclidean(aud3_Energies_teste,medEnerg_N)
DistEucl_aud4_DomTemp_Teste_N = distance.euclidean(aud4_Energies_teste,medEnerg_N)
DistEucl_aud5_DomTemp_Teste_N = distance.euclidean(aud5_Energies_teste,medEnerg_N)
DistEucl_aud6_DomTemp_Teste_N = distance.euclidean(aud6_Energies_teste,medEnerg_N)
DistEucl_aud7_DomTemp_Teste_N = distance.euclidean(aud7_Energies_teste,medEnerg_N)

# -------------------- Calculando a distância Euclidiana para a centroide SIM (dom. Tempo) --------------------
DistEucl_aud1_DomTemp_Teste_S = distance.euclidean(aud1_Energies_teste,medEnerg_S)
DistEucl_aud2_DomTemp_Teste_S = distance.euclidean(aud2_Energies_teste,medEnerg_S)
DistEucl_aud3_DomTemp_Teste_S = distance.euclidean(aud3_Energies_teste,medEnerg_S)
DistEucl_aud4_DomTemp_Teste_S = distance.euclidean(aud4_Energies_teste,medEnerg_S)
DistEucl_aud5_DomTemp_Teste_S = distance.euclidean(aud5_Energies_teste,medEnerg_S)
DistEucl_aud6_DomTemp_Teste_S = distance.euclidean(aud6_Energies_teste,medEnerg_S)
DistEucl_aud7_DomTemp_Teste_S = distance.euclidean(aud7_Energies_teste,medEnerg_S)




# -------------------- CALCULO DISTANCIA EUCLIDEANA (dom. TF): --------------------

# -------------------- Calculando a distância Euclidiana para a centroide NÃO ( dom. Tempo) --------------------
DistEucl_aud1_TransF_N = distance.euclidean(aud1_Energ_Filt,medEnerg_N)
DistEucl_aud2_TransF_N = distance.euclidean(aud2_Energ_Filt,medEnerg_N)
DistEucl_aud3_TransF_N = distance.euclidean(aud3_Energ_Filt,medEnerg_N)
DistEucl_aud4_TransF_N = distance.euclidean(aud4_Energ_Filt,medEnerg_N)
DistEucl_aud5_TransF_N = distance.euclidean(aud5_Energ_Filt,medEnerg_N)
DistEucl_aud6_TransF_N = distance.euclidean(aud6_Energ_Filt,medEnerg_N)
DistEucl_aud7_TransF_N = distance.euclidean(aud7_Energ_Filt,medEnerg_N)


# -------------------- Calculando a distância Euclidiana para a centroide SIM (dom,. Tempo) --------------------
DistEucl_aud1_TransF_S = distance.euclidean(aud1_energ,medEnerg_S)
DistEucl_aud2_TransF_S = distance.euclidean(aud2_energ,medEnerg_S)
DistEucl_aud3_TransF_S = distance.euclidean(aud3_energ,medEnerg_S)
DistEucl_aud4_TransF_S = distance.euclidean(aud4_energ,medEnerg_S)
DistEucl_aud5_TransF_S = distance.euclidean(aud5_energ,medEnerg_S)
DistEucl_aud6_TransF_S = distance.euclidean(aud6_energ,medEnerg_S)
DistEucl_aud7_TransF_S = distance.euclidean(aud7_energ,medEnerg_S)



print(DistEucl_aud1_TransF_N, '-', DistEucl_aud4_TransF_N) #valor baixo - valor alto

print(DistEucl_aud1_TransF_S,'-', DistEucl_aud4_TransF_S) # valor alto - valor baixo