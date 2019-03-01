# Neural Network lab
From basics to advanced about neural network and machine learning. 

## Construindo uma Rede Neural.
Neste diretório colocarei arquivos que descrevem os meus passos no aprendizado e construção de uma rede neural utilizando a linguagem Python. Os conceitos de Rede Neural são, de certa forma, bem simples, envolvendo conceitos básicos de Álgebra Linear e Cálculo Diferencial. Traduzir estes conceitos para uma linguagem de programação apresenta algumas armadilhas, ligadas em parte à sintaxe da linguagem usada. Os arquivos aqui depositados foram escritos de maneira a destacar as minúcias exigidas por uma linguagem tão flexivel como a Python. Além disso, fui acrescentando complexidade a cada arquivo com o intuito de deixar o programa cada vez mais completo para programação em Deep Learning.

### Arquivos programas:

* **Gradient_Descent_Lab.ipynb** - Exercício de aplicação para o Gradiente de Descida (Gradient Descent). Trata-se da utilização de um perceptron para classificar um conjunto de dados bidimensionais. O método do Gradiente é usado para atualizar as matrizes peso e a função sigmóide é utilizada como função de ativação. Este arquivo é uma modificação do arquivo fornecido pela Udacity do curso Engenheiro de Machine Learning.

* **NN01_OneHiddenLayer.py** - Arquivo com um hidden layer, em que se pode escolher o numero de nós deste layer. Este é uma reescrita do exercício da Udacity em que se implementa o Backpropagation (regra da cadeia no cálculo diferencial). O conjunto de dados usado é proveniente da admissão de alunos da Universidade da Califórinia LA, em que os dados de entrada possuem 6 categorias e a saída é binária, o aluno é aceito ou não na Universsidade. O programa é linear, escrita básica para entender como funciona o algoritmo.

* **NN02_TwoHiddenLayer.py** - Um pequeno aperfeiçoamento é feito colocando duas hidden layers, podendo escolher quandos nós cada camada possui. Interessante notar que a programação linear começa a ficar saturada, pois a escrita aumenta muito para acomodar mais variáveis no sistema. Acrescentei o gráfico que apresenta a curva de error do sistema, e para as funções de ativação, utilizei a tangente hiperbólica para as hidden layers e a sigmóide para a sinapse de saída.

* **NN03_nHLayer.py** - Um programa com instruções em série limita muito quando queremos ter a liberdade de construir uma rede com várias camadas com diferentes nós. Tendo base o código anterior, a estrutura das camadas foi colocada em loops, permitindo que o treino e teste sejam feitos a partir da lista definida pelo usuário. Cada elemento da lista define uma camada e o valor define o numero de nós. Para manter o código compartimentado, criei uma lib com as funções para fazer os gráficos. 

### Arquivos auxiliares:

* **functions_lib.py** - biblioteca com as funções de ativação e funções erros. 
* **graphics_lib.py** - biblioteca com as funções para os gráficos de Complexidade do Modelo e Acurácia.
* **data_UCLA.py** - programa que prepara os dados para serem usados na rede neural. Lá se encontram o método usado para normalizar os dados e separá-los em treino e validação do modelo.
* **requirements.txt** - arquivo com as versões das bibliotecas do python utilizadas, caso queira executar o códigos acima. Utilizar o comando `$ pip install -r requirements.txt` para fazer o download das bibliotecas utilizadas. 
