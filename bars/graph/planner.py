from __future__ import annotations
import heapq
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
from .boundary import BoundaryIndex
from .types import BARSGraph
@dataclass
class PlanResult:
    found: bool; node_path: List[int]; edge_path: List[int]; total_cost: float; total_risk: float; total_boundary: float; objective: float; variant: str
    def to_row(self) -> Dict[str, float | int | str]:
        return {'found':int(self.found),'num_subgoals':max(0,len(self.node_path)-2),'num_edges':len(self.edge_path),'total_cost':self.total_cost,'total_risk':self.total_risk,'total_boundary':self.total_boundary,'objective':self.objective,'variant':self.variant}
def _empty(variant: str) -> PlanResult: return PlanResult(False,[],[],float('inf'),float('inf'),float('inf'),float('inf'),variant)
def plan_path(graph: BARSGraph, start_node: int, goal_node: int, variant: str='full_bars', lambda_risk: float=1.0, lambda_boundary: float=1.0, boundary: Optional[BoundaryIndex]=None) -> PlanResult:
    variant=variant.lower()
    if start_node==goal_node: return PlanResult(True,[int(start_node)],[],0.0,0.0,0.0,0.0,variant)
    if variant in {'shortest','gas','tdr_shortest'}: return _node_dijkstra(graph,start_node,goal_node,0.0,variant)
    if variant in {'reachability','bars_lite','risk'}: return _node_dijkstra(graph,start_node,goal_node,lambda_risk,variant)
    if variant in {'boundary','full_bars','bars'}:
        if boundary is None: return _node_dijkstra(graph,start_node,goal_node,lambda_risk,'reachability_no_boundary')
        return _line_graph_dijkstra(graph,start_node,goal_node,lambda_risk,lambda_boundary,boundary,variant)
    raise ValueError(f'Unknown planner variant: {variant}')
def _edge_weight(graph: BARSGraph, eid: int, lambda_risk: float) -> float: return float(graph.cost[eid] + lambda_risk*graph.risk[eid])
def _node_dijkstra(graph: BARSGraph, start_node: int, goal_node: int, lambda_risk: float, variant: str) -> PlanResult:
    out=graph.outgoing_edges(); n=graph.num_nodes; dist=np.full(n,np.inf); prev_node=np.full(n,-1,dtype=np.int64); prev_edge=np.full(n,-1,dtype=np.int64); dist[start_node]=0.0; pq=[(0.0,int(start_node))]
    while pq:
        d,u=heapq.heappop(pq)
        if d!=dist[u]: continue
        if u==goal_node: break
        for eid in out[u]:
            v=int(graph.dst[eid]); nd=d+_edge_weight(graph,int(eid),lambda_risk)
            if nd<dist[v]: dist[v]=nd; prev_node[v]=u; prev_edge[v]=int(eid); heapq.heappush(pq,(nd,v))
    if not np.isfinite(dist[goal_node]): return _empty(variant)
    edges=[]; nodes=[]; cur=int(goal_node); nodes.append(cur)
    while cur!=start_node:
        eid=int(prev_edge[cur]); edges.append(eid); cur=int(prev_node[cur]); nodes.append(cur)
    nodes.reverse(); edges.reverse(); total_cost=float(graph.cost[edges].sum()) if edges else 0.0; total_risk=float(graph.risk[edges].sum()) if edges else 0.0
    return PlanResult(True,nodes,edges,total_cost,total_risk,0.0,float(dist[goal_node]),variant)
def _line_graph_dijkstra(graph: BARSGraph, start_node: int, goal_node: int, lambda_risk: float, lambda_boundary: float, boundary: BoundaryIndex, variant: str) -> PlanResult:
    out=graph.outgoing_edges(); start_edges=out[start_node]
    if len(start_edges)==0: return _empty(variant)
    m=graph.num_edges; dist=np.full(m,np.inf); prev_edge=np.full(m,-1,dtype=np.int64); pq=[]
    for eid in start_edges: eid=int(eid); dist[eid]=_edge_weight(graph,eid,lambda_risk); heapq.heappush(pq,(dist[eid],eid))
    best=-1
    while pq:
        d,eid=heapq.heappop(pq)
        if d!=dist[eid]: continue
        if int(graph.dst[eid])==goal_node: best=eid; break
        for ne in out[int(graph.dst[eid])]:
            ne=int(ne); nd=d+_edge_weight(graph,ne,lambda_risk)+lambda_boundary*boundary.boundary_cost(eid,ne)
            if nd<dist[ne]: dist[ne]=nd; prev_edge[ne]=eid; heapq.heappush(pq,(nd,ne))
    if best<0: return _empty(variant)
    edges=[]; cur=best
    while cur>=0: edges.append(int(cur)); cur=int(prev_edge[cur])
    edges.reverse(); nodes=[int(graph.src[edges[0]])]+[int(graph.dst[e]) for e in edges]; total_boundary=sum(boundary.boundary_cost(int(a),int(b)) for a,b in zip(edges[:-1],edges[1:]))
    return PlanResult(True,nodes,edges,float(graph.cost[edges].sum()),float(graph.risk[edges].sum()),float(total_boundary),float(dist[best]),variant)
def nearest_graph_node(graph: BARSGraph, embedding: np.ndarray) -> int: return int(np.argmin(np.sum((graph.node_embeddings-embedding[None,:])**2,axis=1)))
