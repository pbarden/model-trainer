#!/usr/bin/env python3

import json
import re
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import Counter
from dataclasses import dataclass

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    print("Warning: sentence-transformers not available. Using basic text analysis only.")

@dataclass
class NovelTags:
    title: str
    basic_stats: Dict[str, Any]
    themes: List[str]
    genre_hints: List[str]
    characters: List[str]
    locations: List[str]
    sentiment_profile: Dict[str, float]
    writing_style: Dict[str, Any]
    memory_compatibility: Dict[str, Any]
    scene_types: List[str]
    behaviors: List[str]
    narrative_elements: List[str]
    social_dynamics: List[str]

class SimpleNovelTagger:

    def __init__(self):
        self.encoder = None
        if EMBEDDINGS_AVAILABLE:
            try:
                self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
                print("Loaded sentence transformer for enhanced analysis")
            except Exception as e:
                print(f"Could not load sentence transformer: {e}")

        self.theme_keywords = {
            'romance': ['love', 'heart', 'passion', 'kiss', 'marriage', 'wedding', 'beloved', 'darling'],
            'horror': ['fear', 'terror', 'death', 'blood', 'scream', 'nightmare', 'monster', 'evil'],
            'mystery': ['secret', 'clue', 'detective', 'mystery', 'solve', 'investigation', 'suspect'],
            'adventure': ['journey', 'quest', 'travel', 'explore', 'danger', 'expedition', 'voyage'],
            'fantasy': ['magic', 'wizard', 'dragon', 'spell', 'enchant', 'fairy', 'realm', 'sword'],
            'science_fiction': ['space', 'planet', 'robot', 'future', 'technology', 'alien', 'laser'],
            'historical': ['century', 'ancient', 'historical', 'empire', 'kingdom', 'medieval'],
            'war': ['battle', 'war', 'soldier', 'army', 'fight', 'weapon', 'victory', 'defeat'],
            'supernatural': ['ghost', 'spirit', 'supernatural', 'otherworldly', 'phantom', 'haunted'],
            'crime': ['crime', 'murder', 'police', 'criminal', 'theft', 'law', 'justice', 'court'],
            'scifi': ['spacecraft', 'galaxy', 'universe', 'cosmic', 'stellar', 'interstellar', 'spaceship', 'mars', 'moon', 'orbit', 'radiation', 'atomic', 'laboratory', 'experiment', 'scientist', 'research'],
            'spy': ['agent', 'secret', 'intelligence', 'mission', 'undercover', 'surveillance', 'classified', 'operative', 'espionage', 'infiltrate', 'code', 'cipher', 'embassy', 'diplomat'],
            'thriller': ['chase', 'pursue', 'escape', 'tension', 'suspense', 'urgent', 'deadline', 'trapped', 'conspiracy', 'betrayal', 'danger', 'threat', 'hunt', 'flee', 'racing'],
            'action': ['fight', 'combat', 'violence', 'explosion', 'crash', 'speed', 'fast', 'quick', 'strike', 'attack', 'defend', 'assault', 'force', 'power', 'strength'],
            'space': ['space', 'planet', 'star', 'galaxy', 'universe', 'cosmic', 'orbit', 'rocket', 'asteroid', 'comet', 'satellite', 'nebula', 'solar', 'lunar', 'mars', 'venus'],
            'nautical': ['ship', 'ocean', 'sea', 'sailor', 'captain', 'voyage', 'harbor', 'port', 'anchor', 'sail', 'mast', 'deck', 'crew', 'navigation', 'storm', 'maritime', 'vessel', 'fleet', 'boat', 'submarine'],
            'urban': ['city', 'street', 'building', 'crowd', 'traffic', 'apartment', 'office', 'store', 'restaurant', 'subway', 'skyscraper', 'neighborhood', 'metropolitan', 'downtown', 'district', 'sidewalk', 'plaza'],
            'rural': ['farm', 'village', 'countryside', 'field', 'barn', 'harvest', 'farmer', 'rural', 'meadow', 'cottage', 'pastoral', 'agricultural', 'peaceful', 'quiet', 'simple', 'orchard', 'pasture'],
            'artistic': ['art', 'artist', 'painting', 'music', 'song', 'dance', 'creative', 'beauty', 'aesthetic', 'performance', 'theater', 'opera', 'sculpture', 'gallery', 'canvas', 'stage', 'melody'],
            'political': ['government', 'political', 'power', 'revolution', 'democracy', 'tyranny', 'empire', 'throne', 'crown', 'parliament', 'election', 'ruler', 'diplomat', 'minister', 'senator', 'congress'],
            'medical': ['doctor', 'medicine', 'disease', 'hospital', 'patient', 'cure', 'surgery', 'healing', 'physician', 'nurse', 'treatment', 'diagnosis', 'clinic', 'remedy', 'infection'],
            'educational': ['school', 'university', 'student', 'teacher', 'professor', 'education', 'study', 'learn', 'knowledge', 'library', 'classroom', 'academic', 'scholar', 'lesson', 'curriculum'],
            'psychological': ['mind', 'thought', 'memory', 'consciousness', 'madness', 'insanity', 'psychology', 'mental', 'brain', 'emotion', 'feeling', 'subconscious', 'psyche', 'intellect', 'cognitive'],
            'economic': ['money', 'business', 'trade', 'merchant', 'market', 'economy', 'wealth', 'poor', 'rich', 'commerce', 'profit', 'bank', 'investment', 'financial', 'currency', 'economic', 'trading'],
            'legal': ['court', 'judge', 'lawyer', 'trial', 'justice', 'law', 'legal', 'guilty', 'innocent', 'verdict', 'jury', 'witness', 'evidence', 'attorney', 'prosecution', 'defense', 'lawsuit'],
            'western': ['cowboy', 'sheriff', 'saloon', 'ranch', 'cattle', 'desert', 'frontier', 'outlaw', 'gunfighter', 'wilderness', 'horse', 'wagon', 'gold rush', 'mining', 'prairie', 'settlement'],
            'technological': ['machine', 'technology', 'invention', 'engine', 'mechanical', 'factory', 'industrial', 'steam', 'electricity', 'telegraph', 'innovation', 'engineering', 'apparatus', 'device'],
            'class': ['noble', 'peasant', 'aristocrat', 'servant', 'master', 'slave', 'upper class', 'lower class', 'social', 'society', 'status', 'hierarchy', 'elite', 'commoner', 'gentry', 'nobility']
        }

        self.character_patterns = [
            r'\b([A-Z][a-z]+)\s+(?:said|replied|whispered|shouted|asked|thought|answered)',
            r'"[^"]+",?\s+(?:said|replied)\s+([A-Z][a-z]+)',
            r'\b([A-Z][a-z]+)\s+(?:walked|ran|sat|stood|looked|felt|seemed|appeared|turned)'
        ]

        self.location_patterns = [
            r'\b(?:in|at|near|beside|within)\s+(?:the\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:castle|house|room|street|city|town|forest|mountain|palace|church|hospital|school|university|library|park|garden|bridge|river|lake|ocean|sea|island|valley|hill|desert|field|farm|village|district|quarter|avenue|boulevard|road|lane|alley)',
            r'\bthe\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:of|in|at)'
        ]

    def tag_novel(self, novel_path: Path) -> NovelTags:
        print(f"  Analyzing text...")

        with open(novel_path, 'r', encoding='utf-8') as f:
            text = f.read()

        title = novel_path.stem.replace('_', ' ').title()

        basic_stats = self.extract_basic_stats(text)
        themes = self.detect_themes(text)
        genre_hints = self.classify_genre(text, themes)
        characters = self.find_characters(text)
        locations = self.find_locations(text)
        sentiment_profile = self.analyze_sentiment_profile(text)
        writing_style = self.analyze_writing_style(text)
        memory_compatibility = self.create_memory_compatibility_data(text, characters, locations, themes)

        scene_types = self.detect_scene_types(text)
        behaviors = self.detect_character_behaviors(text)
        narrative_elements = self.detect_narrative_elements(text)
        social_dynamics = self.detect_social_dynamics(text)

        return NovelTags(
            title=title,
            basic_stats=basic_stats,
            themes=themes,
            genre_hints=genre_hints,
            characters=characters,
            locations=locations,
            sentiment_profile=sentiment_profile,
            writing_style=writing_style,
            memory_compatibility=memory_compatibility,
            scene_types=scene_types,
            behaviors=behaviors,
            narrative_elements=narrative_elements,
            social_dynamics=social_dynamics
        )

    def extract_basic_stats(self, text: str) -> Dict[str, Any]:
        words = text.split()
        sentences = text.count('.') + text.count('!') + text.count('?')
        paragraphs = len([p for p in text.split('\n\n') if p.strip()])

        return {
            'word_count': len(words),
            'sentence_count': sentences,
            'paragraph_count': paragraphs,
            'avg_sentence_length': len(words) / max(sentences, 1),
            'avg_paragraph_length': len(words) / max(paragraphs, 1),
            'dialogue_ratio': text.count('"') / len(text),
            'capitalization_ratio': sum(1 for c in text if c.isupper()) / len(text),
            'unique_words': len(set(w.lower() for w in words if w.isalpha())),
            'lexical_diversity': len(set(w.lower() for w in words if w.isalpha())) / len(words)
        }

    def detect_themes(self, text: str) -> List[str]:
        text_lower = text.lower()
        detected_themes = []

        for theme, keywords in self.theme_keywords.items():
            score = sum(text_lower.count(word) for word in keywords)

            word_count = len(text.split())
            normalized_score = score / (word_count / 1000)

            if normalized_score > 0.5:
                detected_themes.append(theme)

        return detected_themes

    def classify_genre(self, text: str, themes: List[str]) -> List[str]:
        genre_hints = []

        if 'romance' in themes:
            genre_hints.append('Romance')
        if 'horror' in themes or 'supernatural' in themes:
            genre_hints.append('Gothic/Horror')
        if 'mystery' in themes or 'crime' in themes:
            genre_hints.append('Mystery/Crime')
        if 'adventure' in themes:
            genre_hints.append('Adventure')
        if 'fantasy' in themes:
            genre_hints.append('Fantasy')
        if 'science_fiction' in themes or 'scifi' in themes or 'space' in themes:
            genre_hints.append('Science Fiction')
        if 'historical' in themes:
            genre_hints.append('Historical Fiction')
        if 'war' in themes:
            genre_hints.append('War/Military')
        if 'spy' in themes:
            genre_hints.append('Spy/Espionage')
        if 'thriller' in themes:
            genre_hints.append('Thriller/Suspense')
        if 'action' in themes:
            genre_hints.append('Action/Adventure')
        if 'nautical' in themes:
            genre_hints.append('Maritime/Naval')
        if 'urban' in themes:
            genre_hints.append('Urban Fiction')
        if 'rural' in themes:
            genre_hints.append('Rural/Pastoral')
        if 'artistic' in themes:
            genre_hints.append('Literary/Artistic')
        if 'political' in themes:
            genre_hints.append('Political Fiction')
        if 'medical' in themes:
            genre_hints.append('Medical Drama')
        if 'educational' in themes:
            genre_hints.append('Campus/Academic')
        if 'psychological' in themes:
            genre_hints.append('Psychological Fiction')
        if 'economic' in themes:
            genre_hints.append('Economic/Commercial')
        if 'legal' in themes:
            genre_hints.append('Legal Drama')
        if 'western' in themes:
            genre_hints.append('Western')
        if 'technological' in themes:
            genre_hints.append('Steampunk/Industrial')
        if 'class' in themes:
            genre_hints.append('Social Drama')

        if not genre_hints:
            genre_hints.append('Literary Fiction')

        return genre_hints

    def find_characters(self, text: str) -> List[str]:
        all_matches = []

        for pattern in self.character_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            all_matches.extend(matches)

        char_counts = Counter(all_matches)

        common_words = {'the', 'and', 'but', 'for', 'you', 'all', 'not', 'one', 'can', 'had', 'her', 'was', 'him', 'his', 'she', 'has', 'how', 'who', 'did', 'get', 'may', 'old', 'see', 'now', 'way', 'its', 'two', 'out', 'day', 'got', 'use', 'man', 'new', 'say', 'men', 'boy', 'try', 'ask', 'own', 'run', 'end', 'why', 'let', 'put', 'too', 'old', 'any', 'ago', 'off', 'far', 'set', 'our', 'out'}

        characters = [char for char, count in char_counts.items()
                     if count >= 3 and char.lower() not in common_words and len(char) > 2]

        return sorted(characters[:15])

    def find_locations(self, text: str) -> List[str]:
        all_matches = []

        for pattern in self.location_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            all_matches.extend(matches)

        location_counts = Counter(all_matches)

        common_words = {'the', 'and', 'but', 'for', 'you', 'all', 'not', 'one', 'can', 'had', 'her', 'was', 'him', 'his', 'she', 'has', 'how', 'who', 'did', 'get', 'may', 'old', 'see', 'now', 'way', 'its', 'two', 'out', 'day', 'got', 'use', 'man', 'new', 'say', 'men', 'boy', 'try', 'ask', 'own', 'run', 'end', 'why', 'let', 'put', 'too', 'old', 'any', 'ago', 'off', 'far', 'set', 'our', 'out'}

        locations = [loc for loc, count in location_counts.items()
                    if count >= 2 and loc.lower() not in common_words and len(loc) > 2]

        return sorted(locations[:10])

    def analyze_sentiment_profile(self, text: str) -> Dict[str, float]:
        positive_words = ['happy', 'joy', 'love', 'beautiful', 'wonderful', 'amazing', 'bright', 'warm', 'pleasant', 'delightful', 'cheerful', 'peaceful', 'gentle', 'kind', 'sweet', 'good', 'great', 'excellent', 'perfect', 'lovely']
        negative_words = ['sad', 'angry', 'fear', 'dark', 'cold', 'terrible', 'horrible', 'death', 'pain', 'cruel', 'evil', 'wicked', 'sinister', 'grim', 'dreadful', 'awful', 'bad', 'wrong', 'hate', 'despair']

        text_lower = text.lower()
        positive_count = sum(text_lower.count(word) for word in positive_words)
        negative_count = sum(text_lower.count(word) for word in negative_words)

        total_emotional = positive_count + negative_count

        if total_emotional == 0:
            return {'positive': 0.5, 'negative': 0.5, 'neutral': 1.0}

        return {
            'positive': positive_count / total_emotional,
            'negative': negative_count / total_emotional,
            'emotional_intensity': total_emotional / len(text.split()) * 1000
        }

    def analyze_writing_style(self, text: str) -> Dict[str, Any]:
        words = text.split()
        sentences = text.count('.') + text.count('!') + text.count('?')

        long_words = [w for w in words if len(w) > 6]
        short_sentences = len([s for s in text.split('.') if len(s.split()) < 10])

        return {
            'avg_word_length': sum(len(w) for w in words) / len(words),
            'complex_word_ratio': len(long_words) / len(words),
            'short_sentence_ratio': short_sentences / max(sentences, 1),
            'exclamation_ratio': text.count('!') / max(sentences, 1),
            'question_ratio': text.count('?') / max(sentences, 1),
            'semicolon_usage': text.count(';') / len(text) * 1000,
            'em_dash_usage': text.count('--') / len(text) * 1000
        }

    def create_memory_compatibility_data(self, text: str, characters: List[str], locations: List[str], themes: List[str]) -> Dict[str, Any]:
        words = text.split()

        return {
            'memory_density_estimate': len(characters) + len(locations) + len(themes),
            'character_memory_potential': len(characters),
            'location_memory_potential': len(locations),
            'theme_memory_potential': len(themes),
            'dialogue_memory_potential': text.count('"') / 2,
            'descriptive_memory_potential': len([w for w in words if len(w) > 8]),
            'optimal_chunk_size': min(max(200, len(words) // 100), 400),
            'suggested_memory_types': {
                'character': len(characters) > 5,
                'location': len(locations) > 3,
                'dialogue': text.count('"') > 100,
                'description': len(themes) > 2,
                'emotion': any(t in ['horror', 'romance', 'supernatural'] for t in themes),
                'theme': len(themes) > 1
            }
        }

    def detect_scene_types(self, text: str) -> List[str]:
        text_lower = text.lower()
        scene_keywords = {
            'dialogue_scene': ['said', 'replied', 'asked', 'whispered', 'shouted', 'conversation', 'talking', 'spoke'],
            'action_scene': ['running', 'fighting', 'chase', 'explosion', 'crash', 'battle', 'attack', 'fled'],
            'romantic_scene': ['kissed', 'embraced', 'caressed', 'lover', 'romantic', 'passion', 'intimate'],
            'death_scene': ['died', 'killed', 'murder', 'corpse', 'funeral', 'burial', 'grave', 'lifeless'],
            'discovery_scene': ['found', 'discovered', 'revealed', 'uncovered', 'noticed', 'realized', 'saw'],
            'travel_scene': ['journey', 'traveled', 'arrived', 'departed', 'road', 'path', 'destination'],
            'conflict_scene': ['argument', 'disagreement', 'confrontation', 'dispute', 'tension', 'hostility'],
            'emotional_scene': ['tears', 'crying', 'joy', 'sorrow', 'grief', 'happiness', 'despair'],
            'indoor_scene': ['room', 'house', 'building', 'interior', 'inside', 'hall', 'chamber'],
            'outdoor_scene': ['outside', 'garden', 'forest', 'field', 'street', 'sky', 'nature'],
            'flashback_scene': ['remembered', 'recalled', 'past', 'years ago', 'once', 'memory', 'before'],
            'dream_scene': ['dream', 'nightmare', 'sleeping', 'asleep', 'unconscious', 'vision']
        }

        detected_scenes = []
        word_count = len(text.split())

        for scene_type, keywords in scene_keywords.items():
            score = sum(text_lower.count(word) for word in keywords)
            normalized_score = score / (word_count / 1000)

            if normalized_score > 0.3:
                detected_scenes.append(scene_type)

        return detected_scenes

    def detect_character_behaviors(self, text: str) -> List[str]:
        text_lower = text.lower()
        behavior_keywords = {
            'heroic': ['brave', 'courageous', 'hero', 'rescued', 'saved', 'protected', 'sacrifice'],
            'villainous': ['evil', 'wicked', 'cruel', 'villain', 'betrayed', 'deceived', 'malicious'],
            'mysterious': ['mysterious', 'secretive', 'hidden', 'concealed', 'enigmatic', 'puzzling'],
            'aggressive': ['angry', 'violent', 'hostile', 'aggressive', 'furious', 'rage', 'attack'],
            'compassionate': ['kind', 'gentle', 'caring', 'compassionate', 'helping', 'comfort'],
            'intelligent': ['clever', 'smart', 'intelligent', 'wise', 'brilliant', 'genius'],
            'deceptive': ['lied', 'deceived', 'tricked', 'betrayed', 'dishonest', 'false'],
            'loyal': ['loyal', 'faithful', 'devoted', 'trustworthy', 'reliable', 'steadfast'],
            'fearful': ['afraid', 'scared', 'terrified', 'frightened', 'anxious', 'worried'],
            'ambitious': ['ambitious', 'determined', 'driven', 'goal', 'achieve', 'success']
        }

        detected_behaviors = []
        word_count = len(text.split())

        for behavior, keywords in behavior_keywords.items():
            score = sum(text_lower.count(word) for word in keywords)
            normalized_score = score / (word_count / 1000)

            if normalized_score > 0.2:
                detected_behaviors.append(behavior)

        return detected_behaviors

    def detect_narrative_elements(self, text: str) -> List[str]:
        text_lower = text.lower()
        narrative_keywords = {
            'first_person': ['i am', 'i was', 'i have', 'i had', 'my', 'myself', 'me'],
            'third_person': ['he was', 'she was', 'they were', 'his', 'her', 'their'],
            'foreshadowing': ['would later', 'little did', 'if only', 'unknown to', 'fate would'],
            'flashback': ['years ago', 'remembered', 'recalled', 'in the past', 'before'],
            'prophecy': ['foretold', 'predicted', 'prophecy', 'destined', 'fate', 'future'],
            'symbolism': ['symbol', 'represented', 'metaphor', 'symbolic', 'meaning'],
            'irony': ['ironic', 'ironically', 'contrary to', 'opposite', 'unexpected'],
            'suspense': ['suddenly', 'unknown', 'mysterious', 'suspense', 'tension'],
            'twist': ['revealed', 'shocking', 'unexpected', 'surprise', 'plot twist'],
            'resolution': ['finally', 'at last', 'conclusion', 'end', 'resolution', 'solved']
        }

        detected_elements = []
        word_count = len(text.split())

        for element, keywords in narrative_keywords.items():
            score = sum(text_lower.count(word) for word in keywords)
            normalized_score = score / (word_count / 1000)

            if normalized_score > 0.15:
                detected_elements.append(element)

        return detected_elements

    def detect_social_dynamics(self, text: str) -> List[str]:
        text_lower = text.lower()
        social_keywords = {
            'family': ['father', 'mother', 'brother', 'sister', 'son', 'daughter', 'family'],
            'friendship': ['friend', 'companion', 'ally', 'trusted', 'friendship', 'bond'],
            'romance': ['love', 'lover', 'beloved', 'romantic', 'marriage', 'wedding'],
            'rivalry': ['rival', 'enemy', 'competitor', 'opponent', 'rivalry', 'competition'],
            'mentorship': ['teacher', 'mentor', 'student', 'apprentice', 'guide', 'learned'],
            'authority': ['king', 'queen', 'lord', 'master', 'commander', 'leader', 'ruler'],
            'rebellion': ['rebel', 'revolt', 'revolution', 'uprising', 'defiance', 'resistance'],
            'hierarchy': ['rank', 'status', 'position', 'class', 'noble', 'peasant', 'servant'],
            'betrayal': ['betrayed', 'betrayal', 'traitor', 'deceived', 'backstabbed'],
            'alliance': ['alliance', 'united', 'together', 'partnership', 'cooperation']
        }

        detected_dynamics = []
        word_count = len(text.split())

        for dynamic, keywords in social_keywords.items():
            score = sum(text_lower.count(word) for word in keywords)
            normalized_score = score / (word_count / 1000)

            if normalized_score > 0.2:
                detected_dynamics.append(dynamic)

        return detected_dynamics

def tag_all_novels():
    tagger = SimpleNovelTagger()
    results = {}
    processed_count = 0

    novels_dir = Path("novels")
    if not novels_dir.exists():
        print("Error: novels directory not found")
        return

    novel_dirs = [d for d in novels_dir.iterdir() if d.is_dir()]
    total = len(novel_dirs)

    print(f"Found {total} novel directories to process")
    start_time = time.time()

    for i, novel_dir in enumerate(novel_dirs):
        print(f"\nProcessing {i+1}/{total}: {novel_dir.name}")

        txt_files = list(novel_dir.glob("*.txt"))
        if not txt_files:
            print(f"  No .txt file found in {novel_dir.name}")
            continue

        try:
            tags = tagger.tag_novel(txt_files[0])

            tag_data = {
                'title': tags.title,
                'basic_stats': tags.basic_stats,
                'themes': tags.themes,
                'genre_hints': tags.genre_hints,
                'characters': tags.characters,
                'locations': tags.locations,
                'sentiment_profile': tags.sentiment_profile,
                'writing_style': tags.writing_style,
                'memory_compatibility': tags.memory_compatibility,
                'tagged_date': time.strftime('%Y-%m-%d %H:%M:%S'),
                'source_file': txt_files[0].name
            }

            tag_file = novel_dir / "tags.json"
            with open(tag_file, 'w', encoding='utf-8') as f:
                json.dump(tag_data, f, indent=2)

            results[novel_dir.name] = tag_data
            processed_count += 1

            print(f"  [OK] Tagged: {len(tags.themes)} themes, {len(tags.characters)} characters, {len(tags.locations)} locations")

        except Exception as e:
            print(f"  [ERROR] Error processing {novel_dir.name}: {e}")

    elapsed_time = time.time() - start_time

    master_index = {
        'corpus_metadata': {
            'total_novels': total,
            'processed_novels': processed_count,
            'processing_time_seconds': elapsed_time,
            'tagged_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'tagger_version': '1.0'
        },
        'novels': results
    }

    with open("corpus_tags.json", 'w', encoding='utf-8') as f:
        json.dump(master_index, f, indent=2)

    print(f"\n{'='*60}")
    print(f"CORPUS TAGGING COMPLETE")
    print(f"{'='*60}")
    print(f"Processed: {processed_count}/{total} novels")
    print(f"Time taken: {elapsed_time/60:.1f} minutes")
    print(f"Files created:")
    print(f"  - {processed_count} individual tags.json files")
    print(f"  - 1 master corpus_tags.json index")

    return master_index

def find_novels_by_criteria(criteria: Dict) -> List[str]:
    try:
        with open("corpus_tags.json", 'r', encoding='utf-8') as f:
            corpus_data = json.load(f)
    except FileNotFoundError:
        print("corpus_tags.json not found. Run tag_all_novels() first.")
        return []

    matches = []
    novels = corpus_data.get('novels', {})

    for novel_id, tags in novels.items():
        match = True

        if 'theme' in criteria:
            if criteria['theme'] not in tags.get('themes', []):
                match = False

        if 'genre' in criteria:
            if criteria['genre'] not in tags.get('genre_hints', []):
                match = False

        if 'min_words' in criteria:
            word_count = tags.get('basic_stats', {}).get('word_count', 0)
            if word_count < criteria['min_words']:
                match = False

        if 'max_words' in criteria:
            word_count = tags.get('basic_stats', {}).get('word_count', 0)
            if word_count > criteria['max_words']:
                match = False

        if 'character' in criteria:
            characters = [c.lower() for c in tags.get('characters', [])]
            if criteria['character'].lower() not in characters:
                match = False

        if match:
            matches.append(novel_id)

    return matches

if __name__ == "__main__":
    print("Model Tea Corpus Tagger")
    print("=" * 40)

    choice = input("\n1. Tag all novels\n2. Search novels\n\nChoice (1-2): ")

    if choice == "1":
        tag_all_novels()
    elif choice == "2":
        print("\nAvailable search criteria:")
        print("  theme: romance, horror, mystery, adventure, fantasy, etc.")
        print("  genre: Romance, Gothic/Horror, Mystery/Crime, etc.")
        print("  min_words: minimum word count")
        print("  max_words: maximum word count")
        print("  character: character name")

        print("\nExample searches:")
        print("  Horror novels: find_novels_by_criteria({'theme': 'horror'})")
        print("  Short novels: find_novels_by_criteria({'max_words': 20000})")
        print("  Adventure stories: find_novels_by_criteria({'theme': 'adventure'})")

        exec(input("\nEnter search command: "))
    else:
        print("Invalid choice")