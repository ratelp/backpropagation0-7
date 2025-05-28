
from utils import arredondar_para_baixo, tangenteHiperbolica
import random

entradas = [0, 1, 2, 3, 4, 5, 6, 7]
saidaDesejada = [
    [0,0,0],
    [0,0,1],
    [0,1,0],
    [0,1,1],
    [1,0,0],
    [1,0,1],
    [1,1,0],
    [1,1,1]
]

neuronios = 7

entradasNormalizadas = [arredondar_para_baixo(i/entradas[-1]) for i in entradas]

pesosEntradaOculta = [random.uniform(-1, 1) for _ in range(neuronios)]

biasOculta = [random.uniform(-1,1) for _ in range(neuronios)]

pesosOcultaSaida = [[random.uniform(-1, 1) for _ in range(neuronios)]for _ in range(3)]

biasSaida = [random.uniform(-1, 1) for _ in range(3)]

taxaAprendizado = 0.1

epocas = 10000

for epoca in range(epocas):
    erroRede = 0
    for i in range(len(entradas)):
        entrada = entradasNormalizadas[i]
        alvo = saidaDesejada[i]

        # Somatório das entradas(valores) da camada entrada -> oculta + função de ativação

        saidasOcultas = []
        net = 0
        for j in range(len(pesosEntradaOculta)):
            net = entrada *  pesosEntradaOculta[j] + biasOculta[j]

            x = tangenteHiperbolica(net)

            saidasOcultas.append(x)

        # Somatório das entradas(valores) da camada oculta -> saida + função de ativação

        saida = []
        for j in range(len(pesosOcultaSaida)):
            net = 0
            for k in range(len(pesosOcultaSaida[j])):
                net += saidasOcultas[k] *  pesosOcultaSaida[j][k] 
                
            net += biasSaida[j]

            x = tangenteHiperbolica(net)

            saida.append(x)
        
        erroSaida = [0] * len(saida)

        # Calculo dos erros da Camada de saida

        for j in range(len(saida)):
            erroSaida[j] = (alvo[j] - saida[j])*(1-(saida[j]**2))

        # Calculo dos erros da Camada oculta

        erroOculta = [0] * len(saidasOcultas)

        for j in range(len(saidasOcultas)):
            somatorio = 0
            for k in range(len(pesosOcultaSaida)):
                somatorio += erroSaida[k] * pesosOcultaSaida[k][j]
            erroOculta[j] = (1-(saidasOcultas[j]**2)) * somatorio
        

        # Atualização dos pesos de oculta -> saida

        for j in range(len(pesosOcultaSaida)):
            for k in range(len(pesosOcultaSaida[j])):
                pesosOcultaSaida[j][k] = pesosOcultaSaida[j][k] + (taxaAprendizado*erroSaida[j]*saidasOcultas[k])
            
            biasSaida[j] = biasSaida[j] + taxaAprendizado * erroSaida[j]
        

        # Atualização dos pesos de entrada -> oculta

        for j in range(len(pesosEntradaOculta)):
            pesosEntradaOculta[j] = pesosEntradaOculta[j] + (taxaAprendizado*erroOculta[j]*entrada)
            
            biasOculta[j] = biasOculta[j] + taxaAprendizado * erroOculta[j]
        
    # Calculo do erro da rede

    for j in range(len(erroSaida)):
        erroRede += erroSaida[j]**2

    erroRede = erroRede/2

    # Exibe erro a cada 1000 épocas
    if epoca % 1000 == 0:
        print(f'Época {epoca}, Erro: {erroRede:.4f}')

print("\nTeste Final:")
for i in range(len(entradas)):
    entrada = entradasNormalizadas[i]
    alvo = saidaDesejada[i]

    # Forward: entrada -> oculta
    saidasOcultas = []
    for j in range(len(pesosEntradaOculta)):
        net = entrada * pesosEntradaOculta[j] + biasOculta[j]
        x = tangenteHiperbolica(net)
        saidasOcultas.append(x)

    # Forward: oculta -> saída
    saida = []
    for j in range(len(pesosOcultaSaida)):
        net = 0
        for k in range(len(pesosOcultaSaida[j])):
            net += saidasOcultas[k] * pesosOcultaSaida[j][k]
        net += biasSaida[j]
        x = tangenteHiperbolica(net)
        saida.append(x)


    # Binariza a saída: se ≥ 0.5 → 1, senão → 0
    saidaBinaria = [1 if v >= 0.5 else 0 for v in saida]

    # Mostra os resultados
    print(f"Entrada: {entradas[i]:.2f} → Saída: {saidaBinaria}, Esperado: {alvo}")
