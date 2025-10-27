import logging
from typing import List
from typing import Dict
from typing import Any
from typing import Optional
from typing import Tuple
from typing import Union
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import networkx as nx
from collections import defaultdict
import asyncio

from .vector_db_client import VectorDBClient
from .embedding_models import EmbeddingManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimilaritySearch:
    def __init__(self, 
                 vector_client: VectorDBClient,
                 embedding_manager: EmbeddingManager,
                 config: Dict[str, Any] = None):
        self.vector_client = vector_client
        self.embedding_manager = embedding_manager
        self.config = config or {}
        
        # SDG interlinkage matrix (from your sdg_interlinks.py)
        self.sdg_interlinkages = self._load_sdg_interlinkages()
        
        # Search configuration
        self.default_similarity_threshold = config.get("similarity_threshold", 0.7)
        self.max_results = config.get("max_results", 100)
        
    def _load_sdg_interlinkages(self) -> Dict[int, List[int]]:
        # Simplified interlinkage mapping - replace with your full data
        interlinkages = {
            1: [2, 3, 4, 6, 8, 10],  # No Poverty links
            2: [1, 3, 4, 5, 6, 8, 12], # Zero Hunger links
            3: [1, 2, 4, 5, 6, 8, 10, 11], # Good Health links
            4: [1, 2, 3, 5, 8, 10, 16], # Quality Education links
            5: [1, 2, 3, 4, 8, 10, 16], # Gender Equality links
            6: [1, 2, 3, 7, 11, 12, 13, 14, 15], # Clean Water links
            7: [1, 8, 9, 11, 12, 13], # Affordable Energy links
            8: [1, 2, 3, 4, 5, 7, 9, 10, 12, 16], # Decent Work links
            9: [7, 8, 11, 12, 17], # Industry Innovation links
            10: [1, 3, 4, 5, 8, 11, 16], # Reduced Inequalities links
            11: [1, 3, 6, 7, 9, 10, 12, 13, 15], # Sustainable Cities links
            12: [2, 6, 7, 8, 9, 11, 13, 14, 15], # Responsible Consumption links
            13: [6, 7, 11, 12, 14, 15], # Climate Action links
            14: [6, 12, 13, 15], # Life Below Water links
            15: [6, 11, 12, 13, 14], # Life on Land links
            16: [1, 4, 5, 8, 10, 17], # Peace Justice links
            17: [9, 16] # Partnerships links
        }
        return interlinkages


    async def search(
        self,
        query: str,
        search_type: str = "general",
        language: str = "en",
        region: Optional[str] = None,
        sdg_goals: Optional[List[int]] = None,
        limit: int = 10
    ) -> Dict[str, Any]:
        
        return await self.semantic_search(
            query=query,
            search_type=search_type,
            language=language,
            region=region,
            sdg_goals=sdg_goals,
            limit=limit
        )

    
    async def semantic_search(self, 
                            query: str,
                            search_type: str = "general",
                            language: str = "en",
                            region: str = None,
                            sdg_goals: List[int] = None,
                            limit: int = 10) -> Dict[str, Any]:
        try:
            # Generate query embedding
            query_embedding = self.embedding_manager.encode(query)
            if len(query_embedding.shape) > 1:
                query_embedding = query_embedding[0]
            
            # Base search parameters
            search_params = {
                "query_vector": query_embedding,
                "limit": limit * 2,  # Get more results for filtering
                "additional_fields": [
                    "title", "summary", "content", "sdg_goals", 
                    "region", "language", "confidence_score", "publication_date"
                ]
            }
            
            # Apply filters based on search type
            where_filter = self._build_search_filter(
                search_type=search_type,
                language=language,
                region=region,
                sdg_goals=sdg_goals
            )
            
            if where_filter:
                search_params["where_filter"] = where_filter
            
            # Execute vector search
            raw_results = self.vector_client.search_similar(**search_params)
            
            # Post-process and rank results
            processed_results = await self._process_search_results(
                raw_results, query, search_type, limit
            )
            
            # Add SDG interlinkage suggestions
            interlinkage_suggestions = self._get_interlinkage_suggestions(processed_results)
            
            return {
                "query": query,
                "search_type": search_type,
                "total_results": len(processed_results),
                "results": processed_results[:limit],
                "interlinkage_suggestions": interlinkage_suggestions,
                "search_metadata": {
                    "language": language,
                    "region": region,
                    "sdg_goals": sdg_goals,
                    "similarity_threshold": self.default_similarity_threshold
                }
            }
            
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            raise
    
    def _build_search_filter(self,
                           search_type: str,
                           language: str = None,
                           region: str = None,
                           sdg_goals: List[int] = None) -> Optional[Dict[str, Any]]:
        """Build Weaviate where filter based on search parameters"""
        filters = []
        
        # Language filter
        if language and language != "all":
            filters.append({
                "operator": "Equal",
                "path": ["language"],
                "valueText": language
            })
        
        # Region filter
        if region and region != "all":
            filters.append({
                "operator": "Equal", 
                "path": ["region"],
                "valueText": region
            })
        
        # SDG goals filter
        if sdg_goals:
            filters.append({
                "operator": "ContainsAny",
                "path": ["sdg_goals"],
                "valueIntArray": sdg_goals
            })
        
        # Search type specific filters
        if search_type == "high_quality":
            filters.append({
                "operator": "GreaterThan",
                "path": ["confidence_score"],
                "valueNumber": 0.8
            })
        elif search_type == "recent":
            # Filter for recent publications (last 2 years)
            filters.append({
                "operator": "GreaterThan",
                "path": ["publication_date"],
                "valueDate": "2022-01-01T00:00:00Z"
            })
        
        # Combine filters
        if not filters:
            return None
        elif len(filters) == 1:
            return filters[0]
        else:
            return {
                "operator": "And",
                "operands": filters
            }
    
    async def _process_search_results(self,
                                    raw_results: List[Dict[str, Any]],
                                    query: str,
                                    search_type: str,
                                    limit: int) -> List[Dict[str, Any]]:
        """Post-process and enhance search results"""
        processed_results = []
        
        for result in raw_results:
            # Extract additional metadata
            additional = result.get("_additional", {})
            certainty = additional.get("certainty", 0.0)
            distance = additional.get("distance", 1.0)
            
            # Skip results below similarity threshold
            if certainty < self.default_similarity_threshold:
                continue
            
            # Enhance result with computed fields
            enhanced_result = {
                **result,
                "similarity_score": certainty,
                "distance": distance,
                "relevance_score": self._compute_relevance_score(result, query, search_type),
                "sdg_coverage": self._analyze_sdg_coverage(result.get("sdg_goals", [])),
                "content_quality": self._assess_content_quality(result)
            }
            
            processed_results.append(enhanced_result)
        
        # Sort by relevance score
        processed_results.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        return processed_results[:limit]
    
    def _compute_relevance_score(self, 
                               result: Dict[str, Any],
                               query: str,
                               search_type: str) -> float:
        """Compute comprehensive relevance score"""
        base_score = result.get("_additional", {}).get("certainty", 0.0)
        
        # Factor in confidence score
        confidence_score = result.get("confidence_score", 0.5)
        confidence_weight = 0.2
        
        # Factor in SDG goal relevance
        sdg_goals = result.get("sdg_goals", [])
        sdg_relevance = len(sdg_goals) / 17.0 if sdg_goals else 0.0  # Normalize by max SDGs
        sdg_weight = 0.2
        
        # Factor in content length (prefer substantial content)
        content_length = len(result.get("content", ""))
        length_score = min(content_length / 10000.0, 1.0)  # Normalize to 10k chars
        length_weight = 0.1
        
        # Search type specific adjustments
        type_bonus = 0.0
        if search_type == "high_quality" and confidence_score > 0.8:
            type_bonus = 0.1
        elif search_type == "comprehensive" and len(sdg_goals) > 2:
            type_bonus = 0.1
        
        relevance_score = (
            base_score * 0.5 +
            confidence_score * confidence_weight +
            sdg_relevance * sdg_weight +
            length_score * length_weight +
            type_bonus
        )
        
        return min(relevance_score, 1.0)
    
    def _analyze_sdg_coverage(self, sdg_goals: List[int]) -> Dict[str, Any]:
        """Analyze SDG goal coverage and interlinkages"""
        if not sdg_goals:
            return {"coverage": 0.0, "interlinkages": [], "primary_goal": None}
        
        primary_goal = sdg_goals[0] if sdg_goals else None
        coverage = len(sdg_goals) / 17.0  # Percentage of SDGs covered
        
        # Find interlinkages
        interlinkages = set()
        for goal in sdg_goals:
            if goal in self.sdg_interlinkages:
                interlinkages.update(self.sdg_interlinkages[goal])
        
        # Remove already covered goals
        interlinkages = list(interlinkages - set(sdg_goals))
        
        return {
            "coverage": coverage,
            "goals_covered": len(sdg_goals),
            "primary_goal": primary_goal,
            "interlinkages": interlinkages[:5],  # Top 5 related goals
            "interconnectivity": len(interlinkages) / 17.0
        }
    
    def _assess_content_quality(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Assess content quality metrics"""
        content = result.get("content", "")
        title = result.get("title", "")
        summary = result.get("summary", "")
        
        # Basic quality metrics
        content_length = len(content)
        has_title = len(title.strip()) > 0
        has_summary = len(summary.strip()) > 0
        has_url = bool(result.get("source_url"))
        
        # Text quality heuristics
        sentence_count = content.count('.') + content.count('!') + content.count('?')
        avg_sentence_length = content_length / max(sentence_count, 1)
        
        quality_score = 0.0
        if has_title: quality_score += 0.2
        if has_summary: quality_score += 0.2  
        if has_url: quality_score += 0.1
        if content_length > 500: quality_score += 0.2
        if 50 <= avg_sentence_length <= 200: quality_score += 0.3  # Optimal sentence length
        
        return {
            "quality_score": quality_score,
            "content_length": content_length,
            "has_metadata": has_title and has_summary,
            "avg_sentence_length": avg_sentence_length,
            "estimated_reading_time": max(1, content_length // 250)  # Words per minute
        }
    
    def _get_interlinkage_suggestions(self, 
                                    results: List[Dict[str, Any]]):
        """Generate SDG interlinkage suggestions based on search results"""
        sdg_frequency = defaultdict(int)
        
        # Count SDG goal frequencies in results
        for result in results:
            for goal in result.get("sdg_goals", []):
                sdg_frequency[goal] += 1
        
        # Generate suggestions for top SDGs
        suggestions = []
        for goal, frequency in sorted(sdg_frequency.items(), key=lambda x: x[1], reverse=True)[:3]:
            interlinkages = self.sdg_interlinkages.get(goal, [])
            if interlinkages:
                suggestions.append({
                    "primary_sdg": goal,
                    "frequency": frequency,
                    "related_sdgs": interlinkages[:5],
                    "suggestion": f"Explore connections between SDG {goal} and related goals: {', '.join(map(str, interlinkages[:3]))}"
                })
        
        return suggestions
    
    async def find_similar_documents(self,
                                   document_id: str,
                                   similarity_threshold: float = 0.7,
                                   limit: int = 10) -> List[Dict[str, Any]]:
        """Find documents similar to a given document"""
        try:
            # Get source document embedding
            # This would require storing document embeddings with IDs
            # For now, implement a placeholder
            
            # In production, you'd:
            # 1. Retrieve document by ID
            # 2. Get its embedding
            # 3. Perform similarity search
            # 4. Return similar documents
            
            logger.warning("find_similar_documents not fully implemented - requires document ID storage")
            return []
            
        except Exception as e:
            logger.error(f"Error finding similar documents: {e}")
            raise
    
    async def cluster_search_results(self,
                                   results: List[Dict[str, Any]],
                                   num_clusters: int = 5) -> Dict[str, Any]:
        """Cluster search results for better organization"""
        try:
            if len(results) < num_clusters:
                return {"clusters": [{"documents": results, "theme": "All Results"}]}
            
            # Extract embeddings (would need to be stored with results)
            # For now, use SDG goals as clustering feature
            
            features = []
            for result in results:
                # Create feature vector based on SDG goals
                feature = [0] * 17
                for goal in result.get("sdg_goals", []):
                    if 1 <= goal <= 17:
                        feature[goal-1] = 1
                features.append(feature)
            
            # Perform clustering
            kmeans = KMeans(n_clusters=min(num_clusters, len(results)), random_state=42)
            cluster_labels = kmeans.fit_predict(features)
            
            # Organize results by cluster
            clusters = defaultdict(list)
            for i, label in enumerate(cluster_labels):
                clusters[label].append(results[i])
            
            # Generate cluster themes
            cluster_results = []
            for cluster_id, docs in clusters.items():
                # Determine theme based on most common SDGs
                sdg_counts = defaultdict(int)
                for doc in docs:
                    for goal in doc.get("sdg_goals", []):
                        sdg_counts[goal] += 1
                
                top_sdgs = sorted(sdg_counts.items(), key=lambda x: x[1], reverse=True)[:2]
                theme = f"SDG {top_sdgs}" if top_sdgs else "Mixed Content"
                if len(top_sdgs) > 1:
                    theme += f" & {top_sdgs}"
                
                cluster_results.append({
                    "cluster_id": cluster_id,
                    "theme": theme,
                    "document_count": len(docs),
                    "documents": docs,
                    "dominant_sdgs": [sdg for sdg, _ in top_sdgs]
                })
            
            return {
                "total_clusters": len(cluster_results),
                "clusters": cluster_results,
                "clustering_method": "SDG-based K-means"
            }
            
        except Exception as e:
            logger.error(f"Error clustering search results: {e}")
            return {"clusters": [{"documents": results, "theme": "Unclustered"}]}
    
    def health_check(self) -> Dict[str, Any]:
        """Health check for similarity search service"""
        try:
            # Test vector client connection
            vector_health = self.vector_client.health_check()
            
            # Test embedding manager
            test_embedding = self.embedding_manager.encode("test query")
            embedding_healthy = len(test_embedding) > 0
            
            return {
                "status": "healthy" if vector_health.get("status") == "healthy" and embedding_healthy else "unhealthy",
                "vector_db": vector_health,
                "embedding_manager": {
                    "status": "healthy" if embedding_healthy else "unhealthy",
                    "available_models": list(self.embedding_manager.models.keys())
                },
                "sdg_interlinkages_loaded": len(self.sdg_interlinkages) == 17
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

class SDGRecommendationEngine:
    def __init__(self, similarity_search: SimilaritySearch):
        self.similarity_search = similarity_search
        self.sdg_weights = self._initialize_sdg_weights()

    async def get_recommendations(
        self,
        user_interests: List[int],
        region: Optional[str] = None,
        language: str = "en",
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        if not user_interests:
            return []
        per_goal = max(1, limit // max(1, len(user_interests)))
        bucket: List[Dict[str, Any]] = []
        seen = set()
        # Query per SDG interest to diversify results
        for sdg in user_interests:
            try:
                res = await self.similarity_search.semantic_search(
                    query=f"SDG {sdg}",
                    search_type="general",
                    language=language,
                    region=region,
                    sdg_goals=[sdg],
                    limit=per_goal
                )
                for item in res.get("results", []):
                    # Build a simple dedup key
                    key = (item.get("source_url") or "", item.get("title") or "")
                    if key not in seen:
                        seen.add(key)
                        bucket.append(item)
            except Exception as e:
                logger.warning(f"Recommendation fetch failed for SDG {sdg}: {e}")
                continue
        # Truncate to limit
        return bucket[:limit]

    def health_check(self) -> Dict[str, Any]:
        return {"status": "healthy"}
    
    def _initialize_sdg_weights(self) -> Dict[int, float]:
        # Weights based on UN priority areas and interconnectedness
        return {
            1: 1.0,   # No Poverty - foundational
            2: 0.95,  # Zero Hunger - critical
            3: 0.9,   # Good Health - essential
            4: 0.85,  # Quality Education - long-term impact
            5: 0.8,   # Gender Equality - cross-cutting
            6: 0.9,   # Clean Water - fundamental
            7: 0.85,  # Affordable Energy - enabling
            8: 0.8,   # Decent Work - economic
            9: 0.75,  # Industry Innovation - development
            10: 0.8,  # Reduced Inequalities - social
            11: 0.7,  # Sustainable Cities - urban
            12: 0.75, # Responsible Consumption - environmental
            13: 1.0,  # Climate Action - urgent priority
            14: 0.8,  # Life Below Water - environmental
            15: 0.8,  # Life on Land - environmental
            16: 0.85, # Peace Justice - governance
            17: 0.9   # Partnerships - enabling
        }
    
    async def recommend_content(self,
                              user_interests: List[int],
                              region: str = None,
                              language: str = "en",
                              limit: int = 10) -> Dict[str, Any]:
        """Generate personalized SDG content recommendations"""
        try:
            recommendations = []
            
            # Get content for user's primary interests
            for sdg_goal in user_interests:
                results = self.vector_client.search_by_sdg_goals(
                    sdg_goals=[sdg_goal],
                    limit=limit // len(user_interests) + 2
                )
                
                # Apply recommendation scoring
                for result in results:
                    score = self._calculate_recommendation_score(
                        result, user_interests, region, language
                    )
                    result["recommendation_score"] = score
                    recommendations.append(result)
            
            # Add interlinkage-based recommendations
            interlinkage_recs = await self._get_interlinkage_recommendations(
                user_interests, region, language, limit // 2
            )
            recommendations.extend(interlinkage_recs)
            
            # Sort and deduplicate
            recommendations.sort(key=lambda x: x["recommendation_score"], reverse=True)
            unique_recommendations = self._deduplicate_recommendations(recommendations)
            
            return {
                "user_interests": user_interests,
                "total_recommendations": len(unique_recommendations),
                "recommendations": unique_recommendations[:limit],
                "recommendation_metadata": {
                    "region": region,
                    "language": language,
                    "interlinkage_based": len(interlinkage_recs),
                    "direct_interest": len(recommendations) - len(interlinkage_recs)
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            raise
    
    def _calculate_recommendation_score(self,
                                      result: Dict[str, Any],
                                      user_interests: List[int],
                                      region: str = None,
                                      language: str = None) -> float:
        """Calculate recommendation score for content"""
        base_score = result.get("confidence_score", 0.5)
        
        # Interest alignment score
        result_sdgs = result.get("sdg_goals", [])
        interest_overlap = len(set(result_sdgs) & set(user_interests))
        interest_score = interest_overlap / max(len(user_interests), 1)
        
        # SDG importance weighting
        sdg_weight = sum(self.sdg_weights.get(goal, 0.5) for goal in result_sdgs) / max(len(result_sdgs), 1)
        
        # Region/language bonus
        region_bonus = 0.1 if result.get("region") == region else 0.0
        language_bonus = 0.1 if result.get("language") == language else 0.0
        
        # Content quality bonus
        quality_bonus = 0.1 if len(result.get("content", "")) > 1000 else 0.0
        
        recommendation_score = (
            base_score * 0.4 +
            interest_score * 0.3 +
            sdg_weight * 0.2 +
            region_bonus + language_bonus + quality_bonus
        )
        
        return min(recommendation_score, 1.0)
    
    async def _get_interlinkage_recommendations(self,
                                             user_interests: List[int],
                                             region: str,
                                             language: str,
                                             limit: int) -> List[Dict[str, Any]]:
        """Get recommendations based on SDG interlinkages"""
        related_sdgs = set()
        
        # Find related SDGs through interlinkages
        for goal in user_interests:
            related_sdgs.update(self.similarity_search.sdg_interlinkages.get(goal, []))
        
        # Remove already interested SDGs
        related_sdgs = list(related_sdgs - set(user_interests))
        
        # Get content for related SDGs
        if related_sdgs:
            results = self.similarity_search.vector_client.search_by_sdg_goals(
                sdg_goals=related_sdgs[:5],  # Top 5 related
                limit=limit
            )
            
            # Mark as interlinkage-based recommendations
            for result in results:
                result["recommendation_type"] = "interlinkage"
                result["recommendation_score"] = self._calculate_recommendation_score(
                    result, user_interests, region, language
                ) * 0.8  # Slight penalty for indirect interest
            
            return results
        
        return []
    
    def _deduplicate_recommendations(self, recommendations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate recommendations based on content similarity"""
        seen_titles = set()
        unique_recs = []
        
        for rec in recommendations:
            title = rec.get("title", "").strip().lower()
            if title and title not in seen_titles:
                seen_titles.add(title)
                unique_recs.append(rec)
        
        return unique_recs
