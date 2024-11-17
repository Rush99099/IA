import heapq
from collections import deque
import networkx as nx
import matplotlib.pyplot as plt

# --- Função para desenhar o grafo ---
def desenhar_grafo(grafo, caminho=None):
    """Desenha o grafo usando NetworkX e destaca o caminho final."""
    G = nx.Graph()

    # Adiciona arestas, pesos e tipos de terreno ao grafo NetworkX
    for origem, destinos in grafo.items():
        for destino, (custo, terreno) in destinos.items():
            G.add_edge(origem, destino, weight=custo, terreno=terreno)

    pos = nx.spring_layout(G)
    plt.figure(figsize=(10, 8))

    # Desenhar o grafo
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_size=700,
        node_color="skyblue",
        font_size=10,
        font_weight="bold",
        edge_color="gray"
    )

    # Destacar o caminho, se fornecido
    if caminho:
        caminho_edges = [(caminho[i], caminho[i + 1]) for i in range(len(caminho) - 1)]
        nx.draw_networkx_edges(G, pos, edgelist=caminho_edges, edge_color="red", width=2)

    # Adiciona os pesos das arestas
    edge_labels = {(u, v): f"{d['weight']} ({d['terreno']})" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    plt.title("Grafo Atualizado com Caminho", fontsize=14)
    plt.show()

# --- Algoritmos de Busca ---



#----------BFS-----------------------
def bfs(grafo, inicio, destino):
    fila = deque([(inicio, [inicio])])
    visitados = set()
    while fila:
        atual, caminho = fila.popleft()
        if atual == destino:
            return caminho
        visitados.add(atual)
        for vizinho in grafo.get(atual, {}):
            if vizinho not in visitados:
                fila.append((vizinho, caminho + [vizinho]))
    return None

#-----------------------BFS-COM RESTRIÇÕES DO DRONE------------------
def verificar_restricoes_drone(propriedades, distancia_atual, max_distancia):
    """Verifica se o drone pode usar uma aresta."""
    if not propriedades.get('aereo', False):  # Verifica acesso aéreo
        return False
    if distancia_atual > max_distancia:  # Verifica alcance do drone
        return False
    return True

def bfs_com_restricoes_drone(grafo, inicio, destino, max_distancia):
    fila = deque([(inicio, [inicio], 0)])  # Adiciona a distância acumulada
    visitados = set()

    while fila:
        atual, caminho, distancia_atual = fila.popleft()
        if atual == destino:
            return caminho
        visitados.add(atual)

        for vizinho, (custo, propriedades) in grafo.get(atual, {}).items():
            if vizinho not in visitados and verificar_restricoes_drone(propriedades, distancia_atual + custo, max_distancia):
                fila.append((vizinho, caminho + [vizinho], distancia_atual + custo))

    return None
#------------------------DFS------------------------
def dfs(grafo, inicio, destino, caminho=None, visitados=None):
    if caminho is None:
        caminho = [inicio]
    if visitados is None:
        visitados = set()
    if inicio == destino:
        return caminho
    visitados.add(inicio)
    for vizinho in grafo.get(inicio, {}):
        if vizinho not in visitados:
            resultado = dfs(grafo, vizinho, destino, caminho + [vizinho], visitados)
            if resultado:
                return resultado
    return None

#-----------------------DFS-COM RESTRIÇÕES DO DRONE------------------

#----------------------BUSCA COM CUSTO UNIFORME------------------------
def busca_custo_uniforme(grafo, inicio, destino):
    fila = [(0, inicio, [inicio])]
    visitados = set()
    while fila:
        custo_atual, atual, caminho = heapq.heappop(fila)
        if atual == destino:
            return caminho, custo_atual
        if atual in visitados:
            continue
        visitados.add(atual)
        for vizinho, (custo, _) in grafo.get(atual, {}).items():
            if vizinho not in visitados:
                heapq.heappush(fila, (custo_atual + custo, vizinho, caminho + [vizinho]))
    return None, float('inf')
#-------------------------------GREEDY-------------------------------------
def algoritmo_greedy(grafo, inicio, destino, heuristica):
    fila = [(heuristica[inicio], inicio, [inicio])]
    visitados = set()
    while fila:
        _, atual, caminho = heapq.heappop(fila)
        if atual == destino:
            return caminho
        if atual in visitados:
            continue
        visitados.add(atual)
        for vizinho, (custo, _) in grafo.get(atual, {}).items():
            heapq.heappush(fila, (heuristica[vizinho], vizinho, caminho + [vizinho]))
    return None
#-----------------------A*----------------------------
#def algoritmo_a_estrela(grafo, inicio, destino, heuristica, restricoes_veiculo):
 #   fila = [(0 + heuristica[inicio], 0, inicio, [inicio])]
  #  visitados = set()
   # while fila:
    #    f, g, atual, caminho = heapq.heappop(fila)
     #   if atual == destino:
      #      return caminho, g
       # if atual in visitados:
        #    continue
        #visitados.add(atual)
        #for vizinho, (custo, terreno) in grafo.get(atual, {}).items():
         #   if terreno not in restricoes_veiculo:
          #      continue
           # novo_g = g + custo
            #f_novo = novo_g + heuristica.get(vizinho, float('inf'))
            #heapq.heappush(fila, (f_novo, novo_g, vizinho, caminho + [vizinho]))
    #return None, float('inf')

def algoritmo_a_estrela(grafo, inicio, destino, heuristica, max_distancia):
    fila = [(0 + heuristica[inicio], 0, inicio, [inicio])]
    visitados = set()

    while fila:
        f, g, atual, caminho = heapq.heappop(fila)
        if atual == destino:
            return caminho, g
        if atual in visitados:
            continue
        visitados.add(atual)

        for vizinho, (custo, propriedades) in grafo.get(atual, {}).items():
            if verificar_restricoes_drone(propriedades, g + custo, max_distancia):
                novo_g = g + custo
                f_novo = novo_g + heuristica.get(vizinho, float('inf'))
                heapq.heappush(fila, (f_novo, novo_g, vizinho, caminho + [vizinho]))

    return None, float('inf')


def update_node(self, n):
    """Atualiza o valor de rhs e reconfigura a fila de prioridade para drones."""
    if n != self.inicio:
        self.rhs[n] = min(
            self.g.get(vizinho, float('inf')) + custo
            for vizinho, (custo, propriedades) in self.grafo.get(n, {}).items()
            if verificar_restricoes_drone(propriedades, self.g.get(vizinho, 0), self.max_distancia)
        )

    self.pq = [(k, v) for k, v in self.pq if v != n]
    heapq.heapify(self.pq)
    if self.g[n] != self.rhs[n]:
        heapq.heappush(self.pq, (self.calculate_key(n), n))




#----------------------------------A* DINAMICO-------------------------------
class AStarDinamico:
    def __init__(self, grafo, inicio, destino, heuristica, restricoes_veiculo):
        self.grafo = grafo
        self.inicio = inicio
        self.destino = destino
        self.heuristica = heuristica
        self.restricoes_veiculo = restricoes_veiculo
        self.g = {n: float('inf') for n in grafo}
        self.g[inicio] = 0
        self.fila = []
        heapq.heappush(self.fila, (self.heuristica[inicio], inicio, [inicio]))

    def compute(self):
        while self.fila:
            _, atual, caminho = heapq.heappop(self.fila)
            if atual == self.destino:
                return caminho, self.g[self.destino]
            for vizinho, (custo, terreno) in self.grafo.get(atual, {}).items():
                if terreno not in self.restricoes_veiculo:
                    continue
                novo_g = self.g[atual] + custo
                if novo_g < self.g[vizinho]:
                    self.g[vizinho] = novo_g
                    f = novo_g + self.heuristica.get(vizinho, float('inf'))
                    heapq.heappush(self.fila, (f, vizinho, caminho + [vizinho]))
        return None, float('inf')
    
# Implementação do LPA*
class LPAStar:
    """Implementação do LPA* (Lifelong Planning A*)."""
    def __init__(self, grafo, inicio, destino, restricoes_veiculo):
        self.grafo = grafo
        self.inicio = inicio
        self.destino = destino
        self.restricoes_veiculo = restricoes_veiculo
        self.g = {n: float('inf') for n in grafo}
        self.rhs = {n: float('inf') for n in grafo}
        self.g[inicio] = 0
        self.rhs[inicio] = 0
        self.pq = []
        heapq.heappush(self.pq, (self.calculate_key(inicio), inicio))

    def calculate_key(self, n):
        """Calcula a prioridade na fila."""
        return (min(self.g[n], self.rhs[n]), min(self.g[n], self.rhs[n]))

    def update_node(self, n):
        """Atualiza o valor de rhs e reconfigura a fila de prioridade."""
        if n != self.inicio:
            self.rhs[n] = min(
                self.g.get(vizinho, float('inf')) + custo
                for vizinho, (custo, terreno) in self.grafo.get(n, {}).items()
                if terreno in self.restricoes_veiculo
            )

        # Remove o nó da fila, se presente
        self.pq = [(k, v) for k, v in self.pq if v != n]
        heapq.heapify(self.pq)

        # Adiciona novamente à fila se necessário
        if self.g[n] != self.rhs[n]:
            heapq.heappush(self.pq, (self.calculate_key(n), n))

    def compute_shortest_path(self):
        """Calcula o caminho de menor custo."""
        while self.pq and (self.pq[0][0] < self.calculate_key(self.destino) or self.g[self.destino] != self.rhs[self.destino]):
            _, atual = heapq.heappop(self.pq)

            if self.g[atual] > self.rhs[atual]:
                self.g[atual] = self.rhs[atual]
                for vizinho, (custo, terreno) in self.grafo.get(atual, {}).items():
                    if terreno in self.restricoes_veiculo:
                        self.update_node(vizinho)
            else:
                self.g[atual] = float('inf')
                self.update_node(atual)
                for vizinho, (custo, terreno) in self.grafo.get(atual, {}).items():
                    if terreno in self.restricoes_veiculo:
                        self.update_node(vizinho)

    def get_path(self):
        """Reconstrói o caminho do início ao destino."""
        atual = self.destino
        caminho = [atual]
        while atual != self.inicio:
            proximo = min(
                (vizinho for vizinho, (custo, terreno) in self.grafo.get(atual, {}).items() if terreno in self.restricoes_veiculo),
                key=lambda v: self.g[v],
                default=None
            )
            if proximo is None:
                return None  # Caminho não encontrado
            caminho.append(proximo)
            atual = proximo
        return caminho[::-1]

# --- Função Principal ---
def main():
    # Grafo inicial
    grafo = {
        'A': {'B': (1, {'terreno': 'terra', 'aereo': True}), 'C': (2, {'terreno': 'montanha', 'aereo': False})},
        'B': {'A': (1, {'terreno': 'terra', 'aereo': True}), 'D': (4, {'terreno': 'água', 'aereo': True})},
        'C': {'A': (2, {'terreno': 'montanha', 'aereo': False}), 'D': (2, {'terreno': 'terra', 'aereo': True}), 'E': (5, {'terreno': 'terra', 'aereo': True})},
        'D': {'B': (4, {'terreno': 'água', 'aereo': True}), 'C': (2, {'terreno': 'terra', 'aereo': True}), 'E': (1, {'terreno': 'terra', 'aereo': False})},
        'E': {'C': (5, {'terreno': 'terra', 'aereo': True}), 'D': (1, {'terreno': 'terra', 'aereo': False})}
    }

    veiculos = {
        'caminhão': {'terra'},
        'jipe': {'terra', 'montanha'},
        'barco': {'água'},
        'drone': {'aereo': True}
    }
    
    heuristica = {'A': 5, 'B': 4, 'C': 3, 'D': 2, 'E': 0}
    
    desenhar_grafo(grafo, None)
    print("Escolha o veículo:")
    for i, veiculo in enumerate(veiculos, start=1):
        print(f"{i}. {veiculo}")
    escolha_veiculo = int(input("Digite o número do veículo: "))
    veiculo = list(veiculos.keys())[escolha_veiculo - 1]
    restricoes_veiculo = veiculos[veiculo]
    max_distancia = 10 if veiculo == 'drone' else float('inf')  # Define o alcance para drones
    
    print("Escolha o algoritmo:")
    print("1. BFS")
    print("2. DFS")
    print("3. UCS")
    print("4. Greedy")
    print("5. A*")
    print("6. A* Dinâmico")
    print("7. LPA*")
    print("8. BFS com Restrições")
    
    escolha = int(input("Digite o número do algoritmo: "))

    inicio, destino = 'A', 'E'

    if escolha == 1:
        caminho = bfs(grafo, inicio, destino)
        print(f"Caminho encontrado: {caminho}")
    elif escolha == 2:
        caminho = dfs(grafo, inicio, destino)
        print(f"Caminho encontrado: {caminho}")
    elif escolha == 3:
        caminho, custo = busca_custo_uniforme(grafo, inicio, destino)
        print(f"Caminho encontrado: {caminho} com custo total de {custo}")
    elif escolha == 4:
        caminho = algoritmo_greedy(grafo, inicio, destino, heuristica)
        print(f"Caminho encontrado: {caminho}")
    elif escolha == 5:
        caminho, custo = algoritmo_a_estrela(grafo, inicio, destino, heuristica, restricoes_veiculo)
        print(f"Caminho encontrado: {caminho} com custo total de {custo}")
    elif escolha == 6:
        astar_dinamico = AStarDinamico(grafo, inicio, destino, heuristica, restricoes_veiculo)
        caminho, custo = astar_dinamico.compute()
        print(f"Caminho encontrado: {caminho} com custo total de {custo}")
    elif escolha == 7:
        lpa = LPAStar(grafo, inicio, destino, restricoes_veiculo)
        lpa.compute_shortest_path()
        caminho = lpa.get_path()
        custo = lpa.g[destino]
        print(f"Caminho encontrado: {caminho} com custo total de {custo}")
    elif escolha == 8:
        caminho = bfs_com_restricoes_drone(grafo, inicio, destino, max_distancia)    
        print(f"Caminho encontrado: {caminho}")
    else:
        print("Opção inválida!")
        return

    #print(f"Caminho encontrado: {caminho} com custo total de {custo}")
    desenhar_grafo(grafo, caminho)

if __name__ == "__main__":
    main()
