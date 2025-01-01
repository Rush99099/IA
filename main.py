import ast
import networkx as nx
import matplotlib.pyplot as plt
from collections import deque
import heapq
import random

def importar_grafo_txt(caminho_arquivo):
    """
    Importa um grafo e zonas a partir de um arquivo de texto.
    Estrutura:
    - Zonas: Nó, Gravidade, População, Acessibilidade, Janela de Tempo
    - Rotas: Origem -> Destino, Distância, Condição, Combustível
    """
    zonas = {}
    grafo = {}
    secao = None

    with open(caminho_arquivo, 'r') as arquivo:
        for linha in arquivo:
            linha = linha.strip()
            if not linha or linha.startswith('#'):
                # Identifica a seção ou ignora linhas vazias/comentários
                if linha.startswith('# Zonas'):
                    secao = 'zonas'
                    
                elif linha.startswith('# Rotas'):
                    secao = 'rotas'
                continue

            if secao == 'zonas':
                # Exemplo: A, alta, 500, {'caminhão': True, 'jipe': True, 'drone': True}, 24
                partes = linha.split(';', maxsplit=4)
                nome = partes[0].strip()
                gravidade = partes[1].strip()
                populacao = int(partes[2].strip())
                # Tente avaliar o dicionário; caso contrário, mostre o erro
                try:
                    acessibilidade = ast.literal_eval(partes[3].strip())
                except (SyntaxError, ValueError) as e:
                    raise ValueError(f"Erro ao processar acessibilidade na linha: {linha}\nErro: {e}")

                janela_tempo = int(partes[4].strip())
                zonas[nome] = {
                    'gravidade': gravidade,
                    'populacao': populacao,
                    'acessibilidade': acessibilidade,
                    'janela_tempo': janela_tempo
                }
            elif secao == 'rotas':
                # Exemplo: A -> B, 10, livre, {'caminhão': 5, 'jipe': 3, 'drone': 1}
                partes = linha.split(';', maxsplit=4)
                origem_destino = partes[0].split('->')
                origem = origem_destino[0].strip()
                destino = origem_destino[1].strip()
                distancia = int(partes[1].strip())
                condicao = partes[2].strip()
                combustivel = ast.literal_eval(partes[3].strip())  # Converte para dict

                # Adiciona ao grafo
                if origem not in grafo:
                    grafo[origem] = {}
                grafo[origem][destino] = {
                    'distancia': distancia,
                    'condicao': condicao,
                    'combustivel': combustivel
                }
                
    return grafo, zonas

def mostrar_grafo(grafo, zonas):
    """
    Mostra o grafo usando NetworkX e Matplotlib.
    - Os nós são rotulados com gravidade e população.
    - As arestas mostram distância e condição.
    """
    G = nx.DiGraph()

    # Adicionar arestas com atributos
    for origem, destinos in grafo.items():
        for destino, atributos in destinos.items():
            G.add_edge(
                origem,
                destino,
                distancia=atributos['distancia'],
                condicao=atributos['condicao']
            )

    # Adicionar nós com atributos
    for zona, atributos in zonas.items():
        G.nodes[zona]['gravidade'] = atributos['gravidade']
        G.nodes[zona]['populacao'] = atributos['populacao']
        G.nodes[zona]['janela_tempo'] = atributos['janela_tempo']

    pos = nx.spring_layout(G)
    plt.figure(figsize=(12, 8))

    # Desenhar nós
    nx.draw(
        G, pos, with_labels=True, node_color="skyblue",
        node_size=1500, font_size=10, font_weight="bold"
    )

    # Adicionar rótulos aos nós
    node_labels = {
        node: f"{node}\n({G.nodes[node]['gravidade']}, {G.nodes[node]['populacao']}p)"
        for node in G.nodes()
    }
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=9)

    # Adicionar rótulos às arestas
    edge_labels = {
        (u, v): f"{d['distancia']}km ({d['condicao']})"
        for u, v, d in G.edges(data=True)
    }
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    plt.title("Grafo com Zonas e Rotas", fontsize=16)
    plt.show()

def desenhar_resultado(grafo, caminho=None, titulo="Resultado do Grafo"):
    """
    Desenha o grafo com destaque no caminho encontrado.
    
    Parâmetros:
        grafo (dict): O grafo no formato {origem: {destino: atributos}}.
        caminho (list): Lista de nós representando o caminho encontrado.
        titulo (str): Título do gráfico.
    """
    G = nx.DiGraph()

    # Adiciona arestas e suas propriedades
    for origem, destinos in grafo.items():
        for destino, atributos in destinos.items():
            G.add_edge(
                origem,
                destino,
                distancia=atributos['distancia'],
                condicao=atributos['condicao']
            )

    # Define as posições para os nós
    pos = nx.spring_layout(G)

    # Configura os nós
    node_colors = []
    for node in G.nodes():
        if caminho and node in caminho:
            node_colors.append("limegreen")  # Nó no caminho destacado
        else:
            node_colors.append("skyblue")   # Nó comum

    # Configura as arestas
    edge_colors = []
    edge_widths = []
    for u, v in G.edges():
        if caminho and u in caminho and v in caminho and caminho.index(u) + 1 == caminho.index(v):
            edge_colors.append("red")  # Aresta no caminho destacado
            edge_widths.append(2.5)    # Aresta destacada
        else:
            edge_colors.append("gray") # Aresta comum
            edge_widths.append(1)      # Aresta padrão

    # Desenhar os nós e arestas
    plt.figure(figsize=(12, 8))
    nx.draw(
        G, pos, with_labels=True, node_color=node_colors,
        node_size=1500, font_size=10, font_weight="bold",
        edge_color=edge_colors, width=edge_widths
    )

    # Adiciona rótulos às arestas
    edge_labels = {
        (u, v): f"{d['distancia']}km ({d['condicao']})"
        for u, v, d in G.edges(data=True)
    }
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    plt.title(titulo, fontsize=16)
    plt.show()


def dfs(grafo, inicio, destino, caminho=None, visitados=None):
    """
    Busca em Profundidade (DFS) adaptada para o grafo fornecido.
    """
    if caminho is None:
        caminho = [inicio]
    if visitados is None:
        visitados = set()
    if inicio == destino:
        return caminho
    visitados.add(inicio)

    for vizinho, atributos in grafo.get(inicio, {}).items():
        if (
            vizinho not in visitados #and 
            #atributos['condicao'] == 'livre' and atributos['condicao'] == 'somente_drones'  # Apenas caminhos livres
        ):
            resultado = dfs(grafo, vizinho, destino, caminho + [vizinho], visitados)
            if resultado:
                return resultado

    return None


def bfs(grafo, inicio, destino):
    """
    Busca em Largura (BFS) adaptada para o grafo fornecido.
    """
    fila = deque([(inicio, [inicio])])  # Cada elemento é (nó_atual, caminho_percorrido)
    visitados = set()

    while fila:
        atual, caminho = fila.popleft()
        if atual == destino:
            return caminho
        visitados.add(atual)

        for vizinho, atributos in grafo.get(atual, {}).items():
            if (
                vizinho not in visitados #and 
                #atributos['condicao'] == 'livre'  # Apenas caminhos livres
            ):
                fila.append((vizinho, caminho + [vizinho]))

    return None


def algoritmo_greedy(grafo, inicio, destino, heuristica):
    """
    Algoritmo Greedy adaptado para o grafo fornecido.
    Usa uma heurística para explorar o caminho com menor custo estimado ao destino.
    """
    fila = [(heuristica[inicio], inicio, [inicio])]  # (h, nó atual, caminho)
    visitados = set()

    while fila:
        h, atual, caminho = heapq.heappop(fila)
        if atual == destino:
            return caminho
        if atual in visitados:
            continue
        visitados.add(atual)

        for vizinho, atributos in grafo.get(atual, {}).items():
           # if atributos['condicao'] == 'livre':  # Apenas caminhos livres
                heapq.heappush(fila, (heuristica[vizinho], vizinho, caminho + [vizinho]))

    return None  # Se nenhum caminho for encontrado


def bfs_com_varias_restricoes(grafo, zonas, inicio, destino, restricoes):
    """
    BFS com restrições que garante um caminho, mesmo que não chegue ao destino.
    """
    fila = deque([(inicio, [inicio], 0)])  # (nó atual, caminho percorrido, combustível usado)
    visitados = set()
    melhor_caminho = None

    while fila:
        atual, caminho, combustivel = fila.popleft()
        visitados.add(atual)

        # Atualizar melhor caminho, mesmo que não chegue ao destino
        if melhor_caminho is None or len(caminho) > len(melhor_caminho):
            melhor_caminho = caminho

        # Verificar se chegou ao destino
        if atual == destino:
            return caminho, combustivel

        for vizinho, atributos in grafo.get(atual, {}).items():
            if (
                vizinho not in visitados and
                atributos['condicao'] in restricoes['condicoes_permitidas'] and
                restricoes['veiculo'] in atributos['combustivel'] and
                combustivel + atributos['combustivel'][restricoes['veiculo']] <= restricoes['combustivel_maximo'] and
                restricoes['veiculo'] in zonas[vizinho]['acessibilidade']
            ):
                fila.append((vizinho, caminho + [vizinho], combustivel + atributos['combustivel'][restricoes['veiculo']]))

    # Retornar o melhor caminho encontrado, mesmo que não chegue ao destino
    return melhor_caminho, combustivel


def dfs_com_varias_restricoes(grafo, zonas, inicio, destino, restricoes, caminho=None, visitados=None, combustivel=0):
    """
    DFS com restrições que garante um caminho, mesmo que não chegue ao destino.
    """
    if caminho is None:
        caminho = [inicio]
    if visitados is None:
        visitados = set()
    visitados.add(inicio)

    # Verificar se chegou ao destino
    if inicio == destino:
        return caminho, combustivel

    melhor_caminho = caminho

    for vizinho, atributos in grafo.get(inicio, {}).items():
        if (
            vizinho not in visitados and
            atributos['condicao'] in restricoes['condicoes_permitidas'] and
            restricoes['veiculo'] in atributos['combustivel'] and
            combustivel + atributos['combustivel'][restricoes['veiculo']] <= restricoes['combustivel_maximo'] and
            restricoes['veiculo'] in zonas[vizinho]['acessibilidade']
        ):
            novo_caminho, novo_combustivel = dfs_com_varias_restricoes(
                grafo, zonas, vizinho, destino, restricoes, caminho + [vizinho], visitados,
                combustivel + atributos['combustivel'][restricoes['veiculo']]
            )
            if novo_caminho and len(novo_caminho) > len(melhor_caminho):
                melhor_caminho = novo_caminho

    # Retornar o melhor caminho encontrado
    return melhor_caminho, combustivel


def algoritmo_greedy_com_restricoes(grafo, zonas, inicio, destino, heuristica, restricoes):
    """
    Greedy com restrições que garante um caminho, mesmo que não chegue ao destino.
    """
    fila = [(heuristica[inicio], inicio, [inicio], 0)]  # (h, nó atual, caminho, combustível usado)
    visitados = set()
    melhor_caminho = None

    while fila:
        h, atual, caminho, combustivel = heapq.heappop(fila)
        visitados.add(atual)

        # Atualizar melhor caminho
        if melhor_caminho is None or len(caminho) > len(melhor_caminho):
            melhor_caminho = caminho

        # Verificar se chegou ao destino
        if atual == destino:
            return caminho, combustivel

        for vizinho, atributos in grafo.get(atual, {}).items():
            if (
                vizinho not in visitados and
                atributos['condicao'] in restricoes['condicoes_permitidas'] and
                restricoes['veiculo'] in atributos['combustivel'] and
                combustivel + atributos['combustivel'][restricoes['veiculo']] <= restricoes['combustivel_maximo'] and
                restricoes['veiculo'] in zonas[vizinho]['acessibilidade']
            ):
                novo_combustivel = combustivel + atributos['combustivel'][restricoes['veiculo']]
                heapq.heappush(fila, (heuristica[vizinho], vizinho, caminho + [vizinho], novo_combustivel))

    # Retornar o melhor caminho encontrado
    return melhor_caminho, combustivel


def algoritmo_a_estrela_com_varias_restricoes(grafo, zonas, inicio, destino, heuristica, restricoes):
    """
    A* com restrições que garante um caminho, mesmo que não chegue ao destino.
    """
    fila = [(heuristica[inicio], 0, inicio, [inicio])]  # (f, g, nó atual, caminho)
    visitados = set()
    melhor_caminho = None

    while fila:
        f, g, atual, caminho = heapq.heappop(fila)
        visitados.add(atual)

        # Atualizar melhor caminho
        if melhor_caminho is None or len(caminho) > len(melhor_caminho):
            melhor_caminho = caminho

        # Verificar se chegou ao destino
        if atual == destino:
            return caminho, g

        for vizinho, atributos in grafo.get(atual, {}).items():
            if (
                vizinho not in visitados and
                atributos['condicao'] in restricoes['condicoes_permitidas'] and
                restricoes['veiculo'] in atributos['combustivel'] and
                g + atributos['combustivel'][restricoes['veiculo']] <= restricoes['combustivel_maximo'] and
                restricoes['veiculo'] in zonas[vizinho]['acessibilidade']
            ):
                novo_g = g + atributos['combustivel'][restricoes['veiculo']]
                f_novo = novo_g + heuristica[vizinho]
                heapq.heappush(fila, (f_novo, novo_g, vizinho, caminho + [vizinho]))

    # Retornar o melhor caminho encontrado, mesmo que não chegue ao destino
    return melhor_caminho, float('inf')

class AStarDinamicoComRestricoes:
    """
    A* Dinâmico com restrições que garante um caminho, mesmo que não chegue ao destino.
    """
    def __init__(self, grafo, zonas, inicio, destino, heuristica, restricoes):
        self.grafo = grafo
        self.zonas = zonas
        self.inicio = inicio
        self.destino = destino
        self.heuristica = heuristica
        self.restricoes = restricoes
        self.g = {n: float('inf') for n in grafo}
        self.g[inicio] = 0
        self.fila = []
        heapq.heappush(self.fila, (self.heuristica[inicio], inicio, [inicio], 0))  # (f, nó atual, caminho, combustível)
        self.melhor_caminho = None

    def compute(self):
        while self.fila:
            _, atual, caminho, combustivel = heapq.heappop(self.fila)

            # Atualizar melhor caminho
            if self.melhor_caminho is None or len(caminho) > len(self.melhor_caminho):
                self.melhor_caminho = caminho

            # Verificar se chegou ao destino
            if atual == self.destino:
                return caminho, combustivel

            for vizinho, atributos in self.grafo.get(atual, {}).items():
                if (
                    atributos['condicao'] in self.restricoes['condicoes_permitidas'] and
                    self.restricoes['veiculo'] in atributos['combustivel'] and
                    combustivel + atributos['combustivel'][self.restricoes['veiculo']] <= self.restricoes['combustivel_maximo'] and
                    self.restricoes['veiculo'] in self.zonas[vizinho]['acessibilidade']
                ):
                    novo_combustivel = combustivel + atributos['combustivel'][self.restricoes['veiculo']]
                    f = novo_combustivel + self.heuristica[vizinho]
                    heapq.heappush(self.fila, (f, vizinho, caminho + [vizinho], novo_combustivel))

        # Retornar o melhor caminho encontrado, mesmo que não chegue ao destino
        return self.melhor_caminho, float('inf')

class LPAStarComRestricoes:
    """
    Implementação do LPA* com restrições que garante um caminho, mesmo que não chegue ao destino.
    """
    def __init__(self, grafo, zonas, inicio, destino, restricoes):
        self.grafo = grafo
        self.zonas = zonas
        self.inicio = inicio
        self.destino = destino
        self.restricoes = restricoes

        # Inicializar as variáveis g e rhs
        self.g = {n: float('inf') for n in grafo}
        self.rhs = {n: float('inf') for n in grafo}
        self.g[inicio] = 0
        self.rhs[inicio] = 0

        # Fila de prioridade
        self.fila = []
        heapq.heappush(self.fila, (self.calculate_key(inicio), inicio))

        # Melhor caminho rastreado
        self.melhor_caminho = None

    def calculate_key(self, n):
        """Calcula a prioridade do nó na fila de prioridade."""
        return (min(self.g[n], self.rhs[n]), min(self.g[n], self.rhs[n]))

    def update_node(self, n):
        """Atualiza o nó com base nas variáveis g e rhs."""
        if n != self.inicio:
            self.rhs[n] = min(
                self.g[v] + atributos['distancia']
                for v, atributos in self.grafo.get(n, {}).items()
                if (
                    atributos['condicao'] in self.restricoes['condicoes_permitidas'] and
                    self.restricoes['veiculo'] in atributos['combustivel']
                )
            )
        # Remover o nó da fila se ele já estiver lá
        self.fila = [(k, v) for k, v in self.fila if v != n]
        heapq.heapify(self.fila)

        # Recolocar o nó na fila se necessário
        if self.g[n] != self.rhs[n]:
            heapq.heappush(self.fila, (self.calculate_key(n), n))

    def compute_shortest_path(self):
        """Calcula o caminho de menor custo."""
        while self.fila and (
            self.fila[0][0] < self.calculate_key(self.destino) or
            self.g[self.destino] != self.rhs[self.destino]
        ):
            _, atual = heapq.heappop(self.fila)

            if self.g[atual] > self.rhs[atual]:
                self.g[atual] = self.rhs[atual]
                for vizinho, atributos in self.grafo.get(atual, {}).items():
                    if (
                        atributos['condicao'] in self.restricoes['condicoes_permitidas'] and
                        self.restricoes['veiculo'] in atributos['combustivel']
                    ):
                        self.update_node(vizinho)
            else:
                self.g[atual] = float('inf')
                self.update_node(atual)
                for vizinho, atributos in self.grafo.get(atual, {}).items():
                    if (
                        atributos['condicao'] in self.restricoes['condicoes_permitidas'] and
                        self.restricoes['veiculo'] in atributos['combustivel']
                    ):
                        self.update_node(vizinho)

    def get_path(self):
        """Reconstrói o caminho do início ao destino."""
        atual = self.destino
        caminho = [atual]
        while atual != self.inicio:
            vizinho_melhor = min(
                (vizinho for vizinho, atributos in self.grafo.get(atual, {}).items()
                 if atributos['condicao'] in self.restricoes['condicoes_permitidas'] and
                    self.restricoes['veiculo'] in atributos['combustivel']),
                key=lambda v: self.g[v],
                default=None
            )
            if vizinho_melhor is None:
                break
            caminho.append(vizinho_melhor)
            atual = vizinho_melhor
        return caminho[::-1] if caminho[-1] == self.inicio else caminho

def busca_custo_uniforme_com_restricoes(grafo, zonas, inicio, destino, restricoes):
    """
    UCS com restrições que garante um caminho, mesmo que não chegue ao destino.
    """
    fila = [(0, inicio, [inicio], 0)]  # (custo acumulado, nó atual, caminho, combustível usado)
    visitados = set()
    melhor_caminho = None

    while fila:
        custo, atual, caminho, combustivel = heapq.heappop(fila)
        visitados.add(atual)

        # Atualizar melhor caminho
        if melhor_caminho is None or len(caminho) > len(melhor_caminho):
            melhor_caminho = caminho

        # Verificar se chegou ao destino
        if atual == destino:
            return caminho, combustivel

        for vizinho, atributos in grafo.get(atual, {}).items():
            if (
                vizinho not in visitados and
                atributos['condicao'] in restricoes['condicoes_permitidas'] and
                restricoes['veiculo'] in atributos['combustivel'] and
                combustivel + atributos['combustivel'][restricoes['veiculo']] <= restricoes['combustivel_maximo'] and
                restricoes['veiculo'] in zonas[vizinho]['acessibilidade']
            ):
                novo_combustivel = combustivel + atributos['combustivel'][restricoes['veiculo']]
                heapq.heappush(fila, (custo + atributos['distancia'], vizinho, caminho + [vizinho], novo_combustivel))

    # Retornar o melhor caminho encontrado, mesmo que não chegue ao destino
    return melhor_caminho, float('inf')

def dinamicas(grafo, zonas):
    print("\n=============================")
    print("    RESTRIÇÕES DINAMICAS       ")
    print("=============================")
    print("1. Sim")
    print("2. Não")
    print("=============================")
    opcao = int(input("Escolha a sua opcao::"))
    
    alteracoes = []  # Lista para armazenar as alterações
    
    if opcao == 1:
        for origem, destinos in grafo.items():
            for destino, atributos in destinos.items():
                # Alterar somente rotas "livre" ou "somente_drones"
                if atributos['condicao'] in ['livre', 'somente_drones']:
                    # Decisão aleatória: bloquear a rota ou alterar o combustível
                    acao = random.choices(
                        ['nenhum', 'bloquear', 'alterar_combustivel'],
                        weights=[0.5, 0.2, 0.3],  # 50% para nenhum, 20% para bloquear, 30% para alterar combustível
                        k=1
                    )[0]

                    if acao == 'bloquear':
                        atributos['condicao'] = 'bloqueada'
                        atributos['combustivel'] = {}  # Remove o consumo de combustível
                        alteracoes.append(f"Rota {origem} -> {destino}: bloqueada")
                    
                    elif acao == 'alterar_combustivel':
                         # Gera uma única percentagem de aumento para toda a rota
                        aumento = random.uniform(1.3, 2.0)
                        detalhes_alteracao = []
                        for veiculo in atributos['combustivel']:
                            novo_valor = int(atributos['combustivel'][veiculo] * aumento)
                            detalhes_alteracao.append(f"{veiculo}: {atributos['combustivel'][veiculo]} -> {novo_valor}")
                            atributos['combustivel'][veiculo] = novo_valor
                        alteracoes.append(f"Rota {origem} -> {destino}: combustível alterado ({', '.join(detalhes_alteracao)})")

        # Imprimir as alterações realizadas
        print("\nAlterações realizadas:")
        if alteracoes:
            for alteracao in alteracoes:
                print(f"  - {alteracao}")
        else:
            print("  Nenhuma alteração foi realizada.")
    
        print("\nGrafo alterado:")
        mostrar_grafo(grafo, zonas)
    
    return grafo

   
def printMenu():
    print("\n=============================")
    print("     ALGORITMOS DE BUSCA       ")
    print("=============================")
    print("1. DFS (Busca em Profundidade)")
    print("2. BFS (Busca em Largura)")
    print("3. Greedy")
    print("4. A*")
    print("5. A* Dinâmico")
    print("6. LPA*")
    print("7. Busca Uniforme")
    print("8. Restrições")
    print("9. Adicionar Nova Rota")
    print("0. Sair")
    print("=============================")


def printMenus2():
    print("\n=============================")
    print("    ALGORITMOS DE BUSCA       ")
    print("=============================")
    print("1. DFS (Busca em Profundidade)")
    print("2. BFS (Busca em Largura)")
    print("3. Greedy")
    print("0. Sair")
    print("=============================")


def main():
    print("\n=============================")
    print("   SISTEMA DE RESOLUÇÃO DE   ")
    print("    PROBLEMAS DE BUSCA        ")
    print("=============================")

    nomeGrafo = input("Insira o nome do ficheiro do grafo: ")
    caminho = nomeGrafo
    grafo, zonas = importar_grafo_txt(caminho)
    mostrar_grafo(grafo, zonas)

    print("\n============================")
    print("        RESTRIÇÕES           ")
    print("=============================")
    print("1. Com Restrições")
    print("2. Sem Restrições")
    print("0. Sair")
    print("=============================")
    restricao = int(input("Escolha a opção: "))
    
    if(restricao == 1):
        while True :
            printMenu()
            escolha = int(input("Escolha o algoritmo: "))
            restricoes = {
                    'veiculo': 'camiao',
                    'combustivel_maximo': 50,
                    'condicoes_permitidas': {'livre'}
                }
            origem='A'
            destino='F'
            
            if(escolha == 8):
                print("\n=============================")
                print("     DIGITE AS RESTRIÇÕES      ")
                print("===============================")
                veiculo = input("Digite o tipo de veículo (camiao, jipe, drone): ").strip().lower()
                combustivel_maximo = int(input("Digite o combustível máximo do veículo: "))
                condicoes_permitidas = input("Digite as condições permitidas (separadas por vírgula, ex: livre, somente_drones): ").strip().split(',')

                # Criar restrições
                restricoes = {
                    'veiculo': veiculo,
                    'combustivel_maximo': combustivel_maximo,
                    'condicoes_permitidas': set(map(str.strip, condicoes_permitidas))
                }
                #origem e destino
                origem = input("Digite a origem: ").strip().upper()
                destino = input("Digite o destino: ").strip().upper()
            if(escolha == 1):
                caminho, combustivel = dfs_com_varias_restricoes(grafo, zonas, origem, destino, restricoes)
                print(f"\nCaminho encontrado (DFS): {caminho}")
                print(f"Combustível usado: {combustivel}")
                desenhar_resultado(grafo, caminho, titulo="Resultado do DFS")
            if(escolha == 2):
                caminho, combustivel = bfs_com_varias_restricoes(grafo, zonas, origem, destino, restricoes)
                print(f"\nCaminho encontrado (BFS): {caminho}")
                print(f"Combustível usado: {combustivel}")
                desenhar_resultado(grafo, caminho, titulo="Resultado do BFS")
            if(escolha == 3):
                heuristica = {'A': 50, 'B': 40, 'C': 30, 'D': 20, 'E': 10, 'F':0}
                caminho, combustivel = algoritmo_greedy_com_restricoes(grafo, zonas, origem, destino, heuristica, restricoes)
                print(f"\nCaminho encontrado (Greedy): {caminho}")
                print(f"Combustível usado: {combustivel}")
                desenhar_resultado(grafo, caminho, titulo="Resultado do Greedy")
            if(escolha == 4):
                caminho, combustivel = algoritmo_a_estrela_com_varias_restricoes(grafo, zonas, origem, destino, heuristica, restricoes)
                print(f"\nCaminho encontrado (A*): {caminho}")
                print(f"Combustível usado: {combustivel}")
                desenhar_resultado(grafo, caminho, titulo="Resultado do A*")
            if(escolha == 5):
                astar_dinamico = AStarDinamicoComRestricoes(grafo, zonas, origem, destino, heuristica, restricoes)
                caminho, custo = astar_dinamico.compute()
                print(f"\nCaminho encontrado (A* Dinâmico): {caminho}")
                print(f"Combustível usado: {custo}")
                desenhar_resultado(grafo, caminho, titulo="Resultado do A* Dinâmico")
            if(escolha == 6):
                lpa = LPAStarComRestricoes(grafo, zonas, origem, destino, restricoes)
                lpa.compute_shortest_path()
                caminho = lpa.get_path()
                print(f"\nCaminho encontrado (LPA*): {caminho}")
                desenhar_resultado(grafo, caminho, titulo="Resultado do LPA*")
            if(escolha == 7):
                caminho, combustivel = busca_custo_uniforme_com_restricoes(grafo, zonas, origem, destino, restricoes)
                print(f"\nCaminho encontrado (UCS): {caminho}")
                print(f"Combustível usado: {combustivel}")
                desenhar_resultado(grafo, caminho, titulo="Resultado UCS")
            if(escolha == 9):
                dinamicas(grafo, zonas)
            if( escolha == 0):
                break
                         
    elif(restricao == 2): 
        while True :
            printMenus2()
            escolha = int(input("Escolha o algoritmo: "))
            inicio, destino = 'A', 'E'
            if escolha == 1:
                caminho = dfs(grafo, inicio, destino)
                print(f"\nCaminho encontrado (DFS): {caminho}")
                desenhar_resultado(grafo, caminho, titulo="Resultado do DFS")
            elif escolha == 2:
                caminho = bfs(grafo, inicio, destino)
                print(f"\nCaminho encontrado (BFS): {caminho}")
                desenhar_resultado(grafo, caminho, titulo="Resultado do BFS")
            elif escolha == 3:
                heuristica = {'A': 50, 'B': 40, 'C': 30, 'D': 20, 'E': 10}
                caminho = algoritmo_greedy(grafo, inicio, destino, heuristica)
                print(f"\nCaminho encontrado (Greedy): {caminho}")
                desenhar_resultado(grafo, caminho, titulo="Resultado do Greedy")
            else:
                print("Opção inválida!")
                break
        
    else:
        print("Fechar Programa")
     
if __name__ == "__main__":
    main()
