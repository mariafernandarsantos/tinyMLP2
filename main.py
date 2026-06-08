import numpy as np
import tensorflow as tf 
import time
import random

NOME_MODELO = "stress_model.tflite" 

interpreter = tf.lite.Interpreter(model_path=NOME_MODELO)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

print(f"Modelo '{NOME_MODELO}' carregado com sucesso!\n")


# simulando sensores
def ler_sensores_simulados(cenario="calmo"):
    if cenario == "crise":
        # Simulando uma pessoa com pico de estresse
        hr = random.uniform(95, 125)
        stress = random.uniform(7, 10)
        sweat = random.uniform(4, 5)
    else:
        # Simulando uma pessoa calma 
        hr = random.uniform(60, 85)
        stress = random.uniform(1, 4)
        sweat = random.uniform(1, 2)
        
    return [hr, stress, sweat]

# loop do dispositivo
TAMANHO_JANELA = 10
buffer_sensores = []

cenario_atual = "calmo"
ciclos = 0

while True:
    try:
        # Alterna o cenário a cada 15 leituras para ver o modelo a reagir!
        if ciclos % 30 == 0:
            cenario_atual = "calmo"
            print("\n>O usuário está calmo")
        elif ciclos % 15 == 0:
            cenario_atual = "crise"
            print("\nO usuário começou a ficar estressado")

        # le os sensores
        leitura_atual = ler_sensores_simulados(cenario=cenario_atual)
        buffer_sensores.append(leitura_atual)
        
        # se o buffer passar do tamanho da janela remove o dado mais antigo
        if len(buffer_sensores) > TAMANHO_JANELA:
            buffer_sensores.pop(0)
            
        # preve depois de completar a janela
        if len(buffer_sensores) == TAMANHO_JANELA:
            # transforma a lista numa matriz
            janela_np = np.array(buffer_sensores)
            
            mean_val = np.mean(janela_np, axis=0)
            std_val = np.std(janela_np, axis=0)
            max_val = np.max(janela_np, axis=0)
            
            features = np.concatenate([mean_val, std_val, max_val])
            
            # formata para a entrada do modelo 
            input_data = np.expand_dims(features, axis=0).astype(np.float32)

            # faz a previsão
            interpreter.set_tensor(input_details['index'], input_data)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details['index'])
            
            # classe com maior pontuação é a prevista
            classe_prevista = np.argmax(output_data)

            nomes_classes = ["Nível Baixo (Verde)", "Nível Moderado (Amarelo)", "Nível Alto / Crise (Vermelho)"]
            
            print(f"BPM Médio: {mean_val[0]:.1f} | Suor Máx: {max_val[2]:.1f} --> Previsão: {nomes_classes[classe_prevista]}")
            
        ciclos += 1
        time.sleep(1) 
        
    except KeyboardInterrupt:
        print("\nFim")
        break