# Repositório - Desafio

O objetivo é escolher variáveis apropriadas para executar uma previsão de precipitação de até 3 dias numa cidade no Brasil.

Escolhendo a cidade de São Paulo que tem uma sazonalidade bem característica de chuvas no verão e seca no inverno (Reboita, 2012).
Com influências espacias como ZCOU/ZCAS trazendo umidade a noroeste, frentes frias vindas de sul, indicando a necessidade de utilizar-se
estações no caminho destes eventos para detectar com antecedência a passagem deles em direção (ou não) da estação estudada.


## v1

Utilizando uma única estação para avaliar o quanto de chuva é possível observar/prever usando apenas informação local.

### Variáveis
Foram utilizadas dados da estação SP - MIRANTE (INMET), com as variáveis em frequência diária agregadas:
- U média
- V média
- T max
- T min
- T amplitude
- e max
- e min
- temporal embedding dia do ano cos
- temporal embedding dia do ano sin

U e V são componentes do vento horizontal  
T é a temperatura observada  
e é a umidade específica calculada a partir de Td, temperatura do ponto de orvalho  

### Modelo | (None, 9) -> [(None + 3, 1), (None + 3, 2)]
Modelo simples, com normalização e abstração inicial com ativação tanh, com objetivo inicial de emular os estados passados entre as células LSTM.
E saída dupla, uma com os resultados numéricos e outra com saída categórica, para avaliar valores e probabilidade.
O número de passos na saída é de N + 3, sendo N o número de passos dados e mais 3 passos de previsão, isto foi feito para treinar o decoder de saída para ensinar o modelo a recriar a precipitação observada e avaliar a qualidade do decoder.

### Conclusões
A utilização do tanh antes de entrar com o LSTM esperando emular as saídas do próprio LSTM é falha, porque as saídas numéricas são regidas por um sigmoid limitadora.
A utilização de dados de vento não são compatíveis com a utilização de uma única estação, é necessário dados espaciais para aplicar advecção ou convolução.

Um única estação é insuficiente para enxergar eventos em transporte no entorno (ZCOU/ZCAS, brisas de sul/sudeste, frentes frias), mas ainda pode identificar sistemas locais como convecção.

#### Training
124/124 [==============================] - 5s 43ms/step  
- loss: 0.3252 - value_loss: 0.3058 - prob_loss: 0.0194  
- val_loss: 0.9265 - val_value_loss: 0.8838 - val_prob_loss: 0.0427  

#### Evaluate test
33/33 [==============================] - 1s 12ms/step 
- loss: 0.9186 - value_loss: 0.8717 - prob_loss: 0.0469  


## v2

Ainda usando uma estação, mas modificando as variáveis utilizadas e incrementando o modelo.

### Variáveis
Foram utilizadas dados da estação SP - MIRANTE (INMET), com as variáveis em frequência diária agregadas:
- T max
- T min
- T mean
- T amplitude
- e max
- e min
- e mean
- temporal embedding dia do ano cos
- temporal embedding dia do ano sin

T é a temperatura observada  
e é a umidade específica calculada a partir de Td, temperatura do ponto de orvalho  

### Modelo | (None, 9) -> [(None + 3, 1), (None + 3, 2)]
Modelo um pouco mais complexo, utilizando uma abstração inicial parecida com o modelo Xception, mas apenas com Dense,a
ao invés de Conv2D e SeparableConv2D (poderia ser uma Conv1D + Dense). Utilização de um LSTM com retorno das sequências
antes do LSTM original, para que as variáveis observadas fiquem com o mesmo padrão que a LSTM original produziria.
Também adicionando uma Dense extra antes das saídas.

### Conclusões
Para as imagens imgs_v2/plot_train2_epoch_95_{00..11}.png pode-se observar o crescimento da previsão de chuva que é
consistente para todos passos (na segunda imagem é indicado pelo crescimento dos últimos 3 dias), indicando que há a
possibilidade do modelo estar 'acertando' os valores abstratos que ocorrem no futuro, porém não está convertendo de
forma correta estes valores para quantidade de chuva.

O modelo enviesa facilmente, porém o erro de validação e teste são consistentes na forma em que erram.

#### Training
124/124 [==============================] - 13s 109ms/step  
- loss: 0.0722 - value_loss: 0.0682 - prob_loss: 0.0040  
- val_loss: 1.0295 - val_value_loss: 0.9336 - val_prob_loss: 0.0959  

#### Evaluate test
33/33 [==============================] - 2s 34ms/step
- loss: 1.1187 - value_loss: 1.0137 - prob_loss: 0.1050


## Referências
Reboita, M. S., et al., 2012. Entendendo o Tempo e o Clima na América do Sul. https://periodicos.sbu.unicamp.br/ojs/index.php/td/article/view/8637425/5139
