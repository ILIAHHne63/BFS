#include <deque>
#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <vector>

const int kConstSize = 99999999;

namespace graph {
template <typename T>
struct DefaultEdge : std::pair<T, T> {
  DefaultEdge(const T& first, const T& second)
      : std::pair<T, T>(first, second) {}
  using BaseClass = std::pair<T, T>;
  const T& Start() const { return BaseClass::first; }
  const T& Finish() const { return BaseClass::second; }
  int Weight() const { return 1; }
};
}  // namespace graph

using namespace graph;
template <typename Vertex = int, typename Edge = DefaultEdge<Vertex>>
class AbstractGraph {
 public:
  using VertexType = Vertex;
  using EdgeType = Edge;

  explicit AbstractGraph(size_t vertices_num, size_t edges_num = 0)
      : vertices_number_(vertices_num), edges_number_(edges_num) {}

  size_t GetVerticesNumber() const { return vertices_number_; }
  size_t GetEdgesNumber() const { return edges_number_; }

  virtual std::vector<Vertex> GetVertices() = 0;

  virtual ~AbstractGraph() = default;

  virtual std::vector<Vertex> GetNeighbours(const Vertex& vertex) = 0;

 protected:
  size_t vertices_number_ = 0;
  size_t edges_number_ = 0;
};

template <typename Vertex = int, typename Edge = DefaultEdge<Vertex>,
          typename Weight = int>
class AdjacencyListGraph : public AbstractGraph<Vertex, Edge> {
  class Iterator;

 public:
  AdjacencyListGraph(size_t vertices_num, const std::vector<Edge>& edges)
      : AbstractGraph<Vertex, Edge>(vertices_num, edges.size()) {
    for (auto& edge : edges) {
      edges_[edge.Start()][edge.Finish()] = edge.Weight();
      edges_[edge.Finish()][edge.Start()] = edge.Weight();
      list_[edge.Start()].push_back(edge.Finish());
      list_[edge.Finish()].push_back(edge.Start());
    }
  }

  std::vector<Vertex> GetVertices() override {
    std::vector<Vertex> vertexes;
    for (const auto& iter : list_) {
      vertexes.emplace_back(iter.first);
    }
    return vertexes;
  }

  std::vector<Vertex> GetNeighbours(const Vertex& vertex) override {
    std::vector<Vertex> neighbours = list_[vertex];
    return neighbours;
  }

  int64_t GetNeighbourValue(const Vertex& vertex_first,
                            const Vertex& vertex_second) {
    return list_[vertex_first][vertex_second];
  }

  Iterator GetIteratorOnNeighbours(const Vertex& vertex) const {
    return Iterator(list_[vertex]);
  }

 private:
  class Iterator {
   public:
    explicit Iterator(const std::unordered_map<Vertex, Vertex>& neighbours)
        : begin_{neighbours.begin()},
          end_{neighbours.end()},
          current_{neighbours.begin()} {}
    // NOLINTNEXTLINE
    typename std::unordered_map<Vertex, Vertex>::const_iterator begin() {
      return begin_;
    }

    // NOLINTNEXTLINE
    typename std::unordered_map<Vertex, Vertex>::const_iterator end() {
      return end_;
    }

    Iterator& operator++() {
      ++current_;
      return *this;
    }

    Iterator& operator++(int) {
      Iterator new_iter = *this;
      ++current_;
      return new_iter;
    }

    Vertex operator*() { return *current_; }

   private:
    typename std::unordered_map<Vertex, Vertex>::const_iterator begin_;
    typename std::unordered_map<Vertex, Vertex>::const_iterator end_;
    typename std::unordered_map<Vertex, Vertex>::const_iterator current_;
  };

  std::unordered_map<Vertex, std::vector<Vertex>> list_;
  std::unordered_map<Vertex, std::unordered_map<Vertex, Weight>> edges_;
};

template <typename Vertex = int, typename Edge = DefaultEdge<Vertex>>
class AdjacencyMatrixGraph : public AbstractGraph<Vertex, Edge> {
  class Iterator;

 public:
  AdjacencyMatrixGraph(size_t vertices_num, const std::vector<Edge>& edges)
      : AbstractGraph<Vertex, Edge>(vertices_num, edges.size()) {
    for (const auto& edge : edges) {
      matrix_[edge.Start()].insert(edge.Finish());
      matrix_[edge.Start()].insert(edge.Finish());
    }
  }

  std::vector<Vertex> GetVertices() override {
    std::vector<Vertex> res;
    for (auto iter = matrix_.begin(); iter != matrix_.end(); ++iter) {
      res.push_back(iter->first);
    }
    return res;
  }

  std::vector<Vertex> GetNeighbours(const Vertex& vertex) const override {
    std::vector<Vertex> neighbours;
    for (const auto& iter : matrix_[vertex]) {
      neighbours.push_back(iter);
    }
    return neighbours;
  }

  Iterator GetIteratorOnNeighbours(const Vertex& vertex) {
    return Iterator(matrix_[vertex]);
  }

 private:
  class Iterator {
   public:
    explicit Iterator(const std::unordered_set<Vertex>& neighbours)
        : begin_{neighbours.begin()},
          end_{neighbours.end()},
          current_{neighbours.begin()} {}
    // NOLINTNEXTLINE
    typename std::unordered_set<Vertex>::const_iterator begin() {
      return begin_;
    }

    // NOLINTNEXTLINE
    typename std::unordered_set<Vertex>::const_iterator End() { return end_; }

    Iterator& operator++() {
      ++current_;
      return *this;
    }

    Iterator& operator++(Iterator& iterator) {
      ++current_;
      return *iterator;
    }

    Vertex operator*() { return *current_; }

   private:
    typename std::unordered_set<Vertex>::const_iterator begin_;
    typename std::unordered_set<Vertex>::const_iterator end_;
    typename std::unordered_set<Vertex>::const_iterator current_;
  };

  std::unordered_map<Vertex, std::unordered_set<Vertex>> matrix_;
};

namespace traverses::visitors {
template <class Vertex, class Edge>
class BfsVisitor {
 public:
  virtual void TreeEdge(const Edge& edge) = 0;
  virtual ~BfsVisitor() = default;
};

template <class Vertex, class Edge>
class AncestorBfsVisitor : BfsVisitor<Vertex, Edge> {
 public:
  virtual void TreeEdge(const Edge& edge) {
    ancestors_[edge.Finish()] = edge.Start();
  }

  virtual ~AncestorBfsVisitor() = default;

 private:
  std::unordered_map<Vertex, Vertex> ancestors_;
};

}  //  namespace traverses::visitors

template <class Vertex, class Edge = DefaultEdge<Vertex>>
class BFS {
 private:
  std::deque<Vertex> deque_;
  std::vector<Vertex> distance_;
  std::vector<bool> visited_;
  std::vector<std::vector<Vertex>> answer_;

 public:
  BFS(Vertex vertex_number) {
    answer_.resize(vertex_number + 1);
    for (int i = 0; i < vertex_number + 1; ++i) {
      visited_.push_back(false);
      distance_.push_back(kConstSize);
    }
  }

  std::vector<Vertex> BfsFunction(
      Vertex start, Vertex end,
      AdjacencyListGraph<Vertex, DefaultEdge<Vertex>> graph,
      int64_t vertex_number,
      traverses::visitors::AncestorBfsVisitor<Vertex, DefaultEdge<Vertex>>
          visitor) {
    deque_.push_back(start);
    distance_[start] = 0;
    for (int64_t i = 0; i < vertex_number; ++i) {
      while (!deque_.empty()) {
        Vertex top = deque_.front();
        deque_.pop_front();
        for (size_t it = 0; it < graph.GetNeighbours(top).size(); ++it) {
          if (distance_[graph.GetNeighbourValue(top, it)] == kConstSize) {
            distance_[graph.GetNeighbourValue(top, it)] = distance_[top] + 1;
            visitor.TreeEdge(
                DefaultEdge<Vertex>(graph.GetNeighbourValue(top, it), top));
            answer_[graph.GetNeighbourValue(top, it)].clear();
            answer_[graph.GetNeighbourValue(top, it)].push_back(top);
            deque_.push_back(graph.GetNeighbourValue(top, it));
          }
        }
      }
    }
    std::vector<Vertex> result;
    if (distance_[end] == kConstSize) {
      return result;
    }
    Vertex vertex_trace_find = end;
    while (vertex_trace_find != start) {
      result.push_back(vertex_trace_find);
      vertex_trace_find = answer_[vertex_trace_find].front();
    }
    result.push_back(start);
    return result;
  }
};

template <typename Vertex = int>
bool Find(Vertex& first, Vertex& second,
          std::vector<DefaultEdge<Vertex>>& edges) {
  std::pair<Vertex, Vertex> edge = {first, second};
  std::pair<Vertex, Vertex> edge_revers = {second, first};
  for (size_t i = 0; i < edges.size(); ++i) {
    if (edges[i] == edge || edges[i] == edge_revers) {
      return false;
    }
  }
  return true;
}

int main() {
  std::ios::sync_with_stdio(false);
  std::cin.tie(0);
  std::cout.tie(0);
  int vertex_number;
  int edges_number;
  int start_trace;
  int end_trace;
  int first_vertex;
  int second_vertex;
  std::vector<DefaultEdge<int>> edges;
  std::cin >> vertex_number;
  std::cin >> edges_number;
  std::cin >> start_trace;
  std::cin >> end_trace;
  for (int i = 0; i < edges_number; ++i) {
    std::cin >> first_vertex;
    std::cin >> second_vertex;
    if (Find(first_vertex, second_vertex, edges)) {
      edges.push_back({first_vertex, second_vertex});
    }
  }
  AdjacencyListGraph<int, DefaultEdge<int>> graph(vertex_number, edges);
  traverses::visitors::AncestorBfsVisitor<int, DefaultEdge<int>> visitor;
  BFS<int, DefaultEdge<int>> answer(vertex_number);
  std::vector<int> result;
  result =
      answer.BfsFunction(start_trace, end_trace, graph, vertex_number, visitor);
  if (result.empty()) {
    std::cout << -1;
    return 0;
  }
  for (size_t i = 0; i < result.size(); ++i) {
    std::cout << result[result.size() - 1 - i] << " ";
  }
  return 0;
}
