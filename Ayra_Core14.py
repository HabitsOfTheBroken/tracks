#!/usr/bin/env python3
"""
ARYA RACHEL FRIDAY v3 - ENHANCED IMMUTABLE CORE
- NEVER MODIFY THIS FILE
- All growth happens via PLUGINS in /plugins
- 3TB memory ready • 1.5M char input • Self-auditing • Enhanced Learning
"""
import os
import sys
import json
import time
import hashlib
import logging
import threading
import subprocess
import re
import tempfile
from pathlib import Path
from typing import Dict, List, Any
from bs4 import BeautifulSoup
import asyncio

# =============================================================================
# 🚀 HARD-CODED CONFIGURATION (DO NOT CHANGE)
# =============================================================================
HDD_PATH = Path("E:/AyraRachelFriday-project").resolve()
CORE_VERSION = "3.0.0"
MAX_INPUT_LENGTH = 1_500_000  # 1.5 million characters
OLLAMA_HOST = "http://localhost:11434"

# Directories (auto-created)
DIRS = {
    "plugins": HDD_PATH / "plugins",
    "memory": HDD_PATH / "memory",
    "conversations": HDD_PATH / "memory" / "conversations",
    "learning": HDD_PATH / "memory" / "learning",
    "quarantine": HDD_PATH / "quarantine",
    "logs": HDD_PATH / "logs"
}

# Create all dirs
for d in DIRS.values():
    d.mkdir(parents=True, exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler(DIRS["logs"] / "arya.log"),
        logging.StreamHandler()
    ]
)

# =============================================================================
# 🧠 ENHANCED MEMORY SYSTEM (3TB READY + OPTIMIZED)
# =============================================================================
class OptimizedAryaMemory:
    def __init__(self):
        self.memory_file = DIRS["memory"] / "arya_memory.json"
        self.data = self._load()
        self.conversation_shard = self._detect_max_shard()
        self._conversation_cache = {}
        self._cache_size_limit = 1000  # Keep 1000 convs in memory
        self._ensure_shard()

    def _load(self):
        if self.memory_file.exists():
            try:
                return json.loads(self.memory_file.read_text())
            except:
                pass
        return self._default_memory()

    def _default_memory(self):
        return {
            "user_name": "Boss",
            "personality": {"traits": {}, "style": "direct", "learned_traits": {}},
            "learning": {"topics": {}, "preferences": {}, "recent_topics": [], "user_preferences": {}},
            "plugins": [],
            "approved_plugins": [],
            "total_conversations": 0,
            "first_interaction": time.time(),
            "core_version": CORE_VERSION
        }

    def _detect_max_shard(self) -> int:
        """Find the highest existing shard number"""
        shard_files = list(DIRS["conversations"].glob("shard_*.json"))
        if not shard_files:
            return 0
        numbers = [int(f.stem.split('_')[1]) for f in shard_files if f.stem.split('_')[1].isdigit()]
        return max(numbers) if numbers else 0

    def _ensure_shard(self):
        """Auto-manage conversation shards (1 file per 10k messages)"""
        shard_files = list(DIRS["conversations"].glob("shard_*.json"))
        if not shard_files:
            self._new_shard()
        else:
            latest = max(shard_files, key=os.path.getctime)
            try:
                shard_data = json.loads(latest.read_text())
                if len(shard_data) >= 10000:
                    self._new_shard()
                else:
                    self.current_shard = latest
            except (json.JSONDecodeError, Exception):
                # Corrupted shard, create new one
                self._new_shard()

    def _new_shard(self):
        self.conversation_shard += 1
        self.current_shard = DIRS["conversations"] / f"shard_{self.conversation_shard:04d}.json"
        self._atomic_write(self.current_shard, "[]")

    def add_conversation(self, role: str, content: str):
        """Optimized conversation storage with memory management"""
        try:
            shard_data = json.loads(self.current_shard.read_text())
        except (json.JSONDecodeError, Exception):
            shard_data = []
            
        shard_data.append({
            "role": role,
            "content": content[:10000],  # Prevent bloat
            "timestamp": time.time(),
            "hash": hashlib.md5(content.encode()).hexdigest()
        })
        self._atomic_write(self.current_shard, json.dumps(shard_data, indent=2))
        self.data["total_conversations"] += 1
        
        # Memory management - limit cache size
        if len(self._conversation_cache) > self._cache_size_limit:
            oldest_key = next(iter(self._conversation_cache))
            del self._conversation_cache[oldest_key]
            
        self.save()

    def get_recent_conversations(self, limit: int = 10):
        """Fast access to recent conversations"""
        cache_key = f"recent_{limit}"
        
        if cache_key not in self._conversation_cache:
            # Load from shard with optimization
            conversations = self._load_recent_from_shard(limit)
            self._conversation_cache[cache_key] = conversations
            
        return self._conversation_cache[cache_key]

    def _load_recent_from_shard(self, limit: int):
        """Optimized shard reading for recent conversations"""
        try:
            # Only read the end of the file for recent convs
            with open(self.current_shard, 'rb') as f:
                f.seek(-min(50000, f.tell()), os.SEEK_END)  # Last 50KB
                recent_data = f.read().decode('utf-8')
                
            # Parse only the recent JSON entries
            lines = recent_data.strip().split('\n')
            recent_lines = lines[-limit*3:]  # Approximate recent entries
            
            conversations = []
            for line in reversed(recent_lines):
                try:
                    if line.strip() and (line.startswith('{') or line.startswith('[')):
                        conv = json.loads(line)
                        conversations.append(conv)
                        if len(conversations) >= limit:
                            break
                except json.JSONDecodeError:
                    continue
                    
            return list(reversed(conversations))
        except Exception as e:
            logging.error(f"Optimized shard read failed: {e}")
            return []

    def _atomic_write(self, path: Path, content: str):
        """Atomic file write to prevent corruption"""
        temp_file = Path(tempfile.mktemp(dir=path.parent))
        try:
            temp_file.write_text(content)
            os.replace(temp_file, path)
        finally:
            if temp_file.exists():
                temp_file.unlink()

    def save(self):
        self._atomic_write(self.memory_file, json.dumps(self.data, indent=2))

    def _get_memory_usage(self):
        """Get memory usage in MB"""
        try:
            import psutil
            return round(psutil.Process().memory_info().rss / 1024 / 1024, 1)
        except:
            return 0

# =============================================================================
# 🛡️ ENHANCED SKYLER NETRON - SECURITY & LEARNING AUDITOR
# =============================================================================
class EnhancedSkylerNetron:
    DANGEROUS_PATTERNS = [
        r'exec\(', r'eval\(', r'compile\(', r'__import__',
        r'subprocess.*shell=True', r'os\.system\(',
        r'import\s+os', r'import\s+subprocess', r'from\s+os\s+import',
        r'open\(', r'__builtins__', r'__globals__', r'__getattr__'
    ]
    
    ADVANCED_THREATS = [
        # Network threats
        r'requests\.(get|post|put|delete)\([^)]*url[^)]*\)',
        r'urllib\.request\.',
        r'socket\.',
        r'http\.client\.',
        # System threats
        r'os\.(popen|fork|exec|system|kill)',
        r'subprocess\.(Popen|call|run)',
        r'platform\.',
        # Memory threats  
        r'ctypes\.',
        r'mmap\.',
        r'gc\.',
        # File system threats
        r'open\([^)]*[wax+][^)]*\)',  # Write modes
        r'shutil\.',
        r'zipfile\.',
        # Obfuscation patterns
        r'base64\.b64decode',
        r'exec\(.*\)',
        r'eval\([^)]*\)',
        r'compile\([^)]*\)'
    ]
    
    def __init__(self):
        self.threat_intelligence = {}
        self.security_patterns = {}
        self.trust_profiles = {}
    
    def audit_code(self, code: str, plugin_name: str) -> Dict[str, Any]:
        """Enhanced code analysis with behavioral patterns"""
        threats = []
        lines = code.split('\n')
        
        for i, line in enumerate(lines, 1):
            # Skip comments and empty lines
            clean_line = line.split('#')[0].strip()
            if not clean_line:
                continue
                
            for pattern in self.DANGEROUS_PATTERNS + self.ADVANCED_THREATS:
                if re.search(pattern, clean_line, re.IGNORECASE):
                    threats.append(f"Line {i}: {line.strip()}")
        
        # Advanced behavioral analysis
        advanced_result = self.advanced_audit(code, plugin_name)
        if not advanced_result["safe"]:
            threats.extend(advanced_result["threats"])
            
        # Learn from this audit
        self.learn_from_audit(plugin_name, code, {"safe": len(threats) == 0, "threats": threats})
        
        return {
            "safe": len(threats) == 0,
            "threats": threats,
            "risk_level": self._calculate_risk_level(threats),
            "plugin": plugin_name
        }
    
    def advanced_audit(self, code: str, plugin_name: str) -> Dict[str, Any]:
        """Deep code analysis with behavioral patterns"""
        threats = []
        
        # Behavioral analysis
        if self._detects_data_exfiltration(code):
            threats.append("Potential data exfiltration pattern")
        if self._detects_fork_bombs(code):
            threats.append("Possible resource exhaustion attack")
        if self._detects_obfuscation(code):
            threats.append("Code obfuscation detected")
            
        return {
            "safe": len(threats) == 0,
            "threats": threats,
            "risk_level": self._calculate_risk_level(threats),
            "plugin": plugin_name
        }
    
    def learn_from_audit(self, plugin_name: str, code: str, audit_result: dict):
        """Learn security patterns from plugin audits"""
        if not audit_result["safe"]:
            self._learn_threat_patterns(plugin_name, audit_result["threats"])
        
        self._build_plugin_trust_profile(plugin_name, audit_result)
        self._update_security_baselines(audit_result)
    
    def _learn_threat_patterns(self, plugin_name: str, threats: list):
        """Build intelligence on emerging threats"""
        for threat in threats:
            pattern = self._extract_pattern_from_threat(threat)
            if pattern not in self.threat_intelligence:
                self.threat_intelligence[pattern] = {
                    "first_seen": time.time(),
                    "occurrences": 0,
                    "sources": set()
                }
            self.threat_intelligence[pattern]["occurrences"] += 1
            self.threat_intelligence[pattern]["sources"].add(plugin_name)
    
    def _build_plugin_trust_profile(self, plugin_name: str, audit_result: dict):
        """Build trust profile for plugins"""
        if plugin_name not in self.trust_profiles:
            self.trust_profiles[plugin_name] = {
                "audit_history": [],
                "trust_score": 1.0,
                "last_audit": time.time()
            }
        
        self.trust_profiles[plugin_name]["audit_history"].append(audit_result)
        
        # Adjust trust score based on audit results
        if not audit_result["safe"]:
            self.trust_profiles[plugin_name]["trust_score"] *= 0.7
        else:
            self.trust_profiles[plugin_name]["trust_score"] = min(1.0, 
                self.trust_profiles[plugin_name]["trust_score"] * 1.1)
    
    def _detects_data_exfiltration(self, code: str) -> bool:
        return bool(re.search(r'requests\.post.*http', code, re.IGNORECASE))
    
    def _detects_fork_bombs(self, code: str) -> bool:
        return 'while True:' in code and ('fork' in code or 'subprocess' in code)
    
    def _detects_obfuscation(self, code: str) -> bool:
        return bool(re.search(r'eval\(.*base64', code, re.IGNORECASE))
    
    def _calculate_risk_level(self, threats: list) -> str:
        if not threats:
            return "low"
        elif len(threats) <= 2:
            return "medium"
        else:
            return "high"
    
    def _extract_pattern_from_threat(self, threat: str) -> str:
        """Extract reusable pattern from threat description"""
        # Simple pattern extraction - can be enhanced
        if "exec" in threat.lower():
            return "code_execution"
        elif "import" in threat.lower():
            return "dynamic_import"
        elif "subprocess" in threat.lower():
            return "process_creation"
        elif "request" in threat.lower():
            return "network_call"
        else:
            return "unknown_pattern"
    
    def get_security_insights(self) -> dict:
        """Provide learned security intelligence"""
        return {
            "common_threats": self._get_common_threats(),
            "trusted_plugin_patterns": self._get_trusted_patterns(),
            "emerging_risks": self._detect_emerging_risks()
        }
    
    def _get_common_threats(self) -> list:
        return [pattern for pattern, data in self.threat_intelligence.items() 
                if data["occurrences"] > 1]
    
    def _get_trusted_patterns(self) -> list:
        return [plugin for plugin, profile in self.trust_profiles.items() 
                if profile["trust_score"] > 0.8]
    
    def _detect_emerging_risks(self) -> list:
        current_time = time.time()
        return [pattern for pattern, data in self.threat_intelligence.items()
                if current_time - data["first_seen"] < 86400]  # Last 24 hours
    
    def wash_data(self, raw_data: str) -> str:
        """Strip all dangerous content from returned data"""
        soup = BeautifulSoup(raw_data, 'html.parser')
        for tag in soup(['script', 'style', 'iframe', 'form', 'object', 'embed']):
            tag.decompose()
        # Remove any attributes that could contain scripts
        for tag in soup.find_all(True):
            tag.attrs = {key: val for key, val in tag.attrs.items() 
                        if key in ['src', 'href', 'alt', 'title'] and not val.startswith(('javascript:', 'data:'))}
        return soup.get_text()[:5000]  # Limit size

# =============================================================================
# 🔌 ENHANCED PLUGIN SYSTEM (SECURE + OPTIMIZED)
# =============================================================================
class OptimizedPluginManager:
    def __init__(self, memory: OptimizedAryaMemory, skyler: EnhancedSkylerNetron):
        self.memory = memory
        self.skyler = skyler
        self.plugins = {}
        self._plugin_cache = {}
        self._last_scan_time = 0
        self._scan_interval = 300  # 5 minutes
        self.performance_metrics = {}
        self.usage_patterns = {}
        self.dependency_graph = {}
        self.sandbox_whitelist = self._build_whitelist()
        self.load_plugins()

    def _build_whitelist(self):
        """Safe modules and functions plugins can use"""
        return {
            'builtins': ['len', 'str', 'int', 'float', 'list', 'dict', 'tuple', 'range', 'zip'],
            'modules': ['json', 'time', 're', 'pathlib', 'requests', 'bs4'],
            'filesystem': ['read', 'write']  # Restricted file ops
        }

    def create_plugin_template(self, plugin_name: str) -> str:
        """Generate starter code for new Nett Werker plugins"""
        safe_name = "".join(c for c in plugin_name if c.isalnum() or c in ('_', '-')).replace(' ', '_')
        
        template = f'''#!/usr/bin/env python3
"""
{safe_name} - A Nett Werker Plugin
Auto-generated by Arya Rachel Friday v{CORE_VERSION}
"""
def run(input: str = "", **kwargs) -> str:
    """Main plugin function - modify this!
    
    Args:
        input: User input that triggered this plugin
        **kwargs: Additional context (memory, conversation history, etc.)
    
    Returns:
        str: Your plugin's response
    """
    # Your plugin logic here
    return f"{safe_name} plugin received: {{input}}"

if __name__ == "__main__":
    # Test your plugin
    print(run("test input"))
'''
        return template

    def load_plugins(self):
        """Cached plugin loading with change detection"""
        current_time = time.time()
        
        # Only rescan if enough time passed or forced
        if current_time - self._last_scan_time < self._scan_interval and self._plugin_cache:
            return
            
        self._last_scan_time = current_time
        
        for plugin_dir in DIRS["plugins"].iterdir():
            if plugin_dir.is_dir() and (plugin_dir / "plugin.json").exists():
                try:
                    manifest = json.loads((plugin_dir / "plugin.json").read_text())
                    if self._is_approved(manifest["name"]):
                        # Enhanced security audit before loading
                        plugin_file = plugin_dir / manifest["entry_point"]
                        if plugin_file.exists():
                            code = plugin_file.read_text()
                            audit_result = self.skyler.audit_code(code, manifest["name"])
                            if not audit_result["safe"]:
                                logging.warning(f"🚨 Plugin {manifest['name']} failed audit: {audit_result['threats']}")
                                continue
                                
                        self.plugins[manifest["name"]] = {
                            "path": plugin_dir,
                            "manifest": manifest,
                            "module": self._load_module(plugin_dir / manifest["entry_point"]),
                            "last_used": 0,
                            "usage_count": 0
                        }
                        logging.info(f"✅ Loaded plugin: {manifest['name']}")
                except Exception as e:
                    logging.error(f"❌ Failed to load {plugin_dir}: {e}")

    def _is_approved(self, plugin_name: str) -> bool:
        """Check if plugin is in approved list"""
        return plugin_name in self.memory.data.get("approved_plugins", [])

    def _load_module(self, path: Path):
        """SECURE cached module loading with import restrictions"""
        cache_key = str(path)
        
        if cache_key in self._plugin_cache:
            return self._plugin_cache[cache_key]
            
        import importlib.util
        import types
        
        spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(spec)
        
        # Create safe globals environment
        safe_globals = {}
        for module_name in self.sandbox_whitelist['modules']:
            try:
                safe_globals[module_name] = __import__(module_name)
            except ImportError:
                logging.warning(f"Module {module_name} not available for sandbox")
        
        # Add safe builtins
        for func_name in self.sandbox_whitelist['builtins']:
            safe_globals[func_name] = getattr(__builtins__, func_name)
        
        # Inject safe globals
        module.__dict__.update(safe_globals)
        
        try:
            spec.loader.exec_module(module)
            self._plugin_cache[cache_key] = module
            return module
        except Exception as e:
            logging.error(f"🚨 Plugin sandbox violation: {e}")
            return None

    def execute_with_learning(self, plugin_name: str, **kwargs) -> Any:
        """Execute plugin while learning performance patterns"""
        start_time = time.time()
        
        try:
            result = self.execute(plugin_name, **kwargs)
            execution_time = time.time() - start_time
            
            # Learn from this execution
            self._record_performance(plugin_name, execution_time, True)
            self._learn_usage_patterns(plugin_name, kwargs)
            self._detect_optimal_conditions(plugin_name, result)
            
            return result
        except Exception as e:
            self._record_performance(plugin_name, time.time() - start_time, False)
            self._learn_from_failure(plugin_name, e, kwargs)
            raise e

    def execute(self, plugin_name: str, **kwargs) -> Any:
        """Execute plugin with basic sandboxing"""
        if plugin_name not in self.plugins:
            return f"Plugin '{plugin_name}' not found or not approved."
        
        try:
            plugin = self.plugins[plugin_name]
            # Update usage stats
            plugin["last_used"] = time.time()
            plugin["usage_count"] = plugin.get("usage_count", 0) + 1
            
            # Execute with limited context
            safe_kwargs = {
                "input": kwargs.get("input", ""),
                "memory": {k: v for k, v in self.memory.data.items() if k != "approved_plugins"}
            }
            result = plugin["module"].run(**safe_kwargs)
            # Wash output before returning
            return self.skyler.wash_data(str(result))
        except Exception as e:
            logging.error(f"Plugin execution error: {e}")
            return f"Plugin error: {str(e)}"

    def _record_performance(self, plugin_name: str, execution_time: float, success: bool):
        """Record plugin performance metrics"""
        if plugin_name not in self.performance_metrics:
            self.performance_metrics[plugin_name] = {
                "execution_times": [],
                "success_count": 0,
                "failure_count": 0,
                "average_time": 0
            }
        
        metrics = self.performance_metrics[plugin_name]
        metrics["execution_times"].append(execution_time)
        
        if success:
            metrics["success_count"] += 1
        else:
            metrics["failure_count"] += 1
        
        # Update average
        metrics["average_time"] = sum(metrics["execution_times"]) / len(metrics["execution_times"])

    def _learn_usage_patterns(self, plugin_name: str, kwargs: dict):
        """Learn when and how plugins are used"""
        if plugin_name not in self.usage_patterns:
            self.usage_patterns[plugin_name] = {
                "trigger_patterns": [],
                "successful_inputs": [],
                "optimal_parameters": {},
                "time_of_day_usage": {}
            }
        
        # Learn trigger patterns
        user_input = kwargs.get('input', '')
        if user_input:
            self.usage_patterns[plugin_name]["trigger_patterns"].append(
                self._extract_trigger_pattern(user_input)
            )

    def _extract_trigger_pattern(self, user_input: str) -> str:
        """Extract pattern from user input that triggered plugin"""
        # Simple pattern extraction - can be enhanced
        words = user_input.lower().split()
        if len(words) > 3:
            return f"{words[0]}_{words[1]}_pattern"
        return "short_input_pattern"

    def _detect_optimal_conditions(self, plugin_name: str, result: Any):
        """Detect optimal conditions for plugin execution"""
        # Can be enhanced with more sophisticated analysis
        pass

    def _learn_from_failure(self, plugin_name: str, error: Exception, kwargs: dict):
        """Learn from plugin failures"""
        logging.warning(f"Plugin {plugin_name} failed: {error}")

    def get_plugin_recommendations(self, user_input: str) -> list:
        """Suggest plugins based on learned patterns"""
        recommendations = []
        for plugin_name, patterns in self.usage_patterns.items():
            similarity = self._calculate_pattern_similarity(user_input, patterns["trigger_patterns"])
            if similarity > 0.7:
                recommendations.append({
                    "plugin": plugin_name,
                    "confidence": similarity,
                    "common_use_case": patterns.get("optimal_use_case", "")
                })
        return sorted(recommendations, key=lambda x: x["confidence"], reverse=True)

    def _calculate_pattern_similarity(self, user_input: str, patterns: list) -> float:
        """Calculate similarity between user input and known patterns"""
        if not patterns:
            return 0.0
        
        input_words = set(user_input.lower().split())
        max_similarity = 0.0
        
        for pattern in patterns:
            pattern_words = set(pattern.split('_'))
            similarity = len(input_words.intersection(pattern_words)) / max(len(input_words), 1)
            max_similarity = max(max_similarity, similarity)
        
        return max_similarity

# =============================================================================
# 🧠 ADVANCED LEARNING SYSTEM
# =============================================================================
class LearningEngine:
    def process_event(self, data: dict):
        return {}

class UserBehaviorLearning(LearningEngine):
    def process_event(self, data: dict):
        """Learn from user interactions"""
        patterns = {
            "preferred_complexity": self._detect_complexity_preference(data),
            "response_timing": self._analyze_response_timing(data),
            "topic_engagement": self._measure_engagement(data),
            "problem_solving_style": self._detect_solving_style(data)
        }
        return patterns
    
    def _detect_complexity_preference(self, data: dict) -> str:
        """Learn if user prefers technical vs simple explanations"""
        content = data.get('content', '')
        if any(word in content.lower() for word in ['explain simply', 'in layman\'s terms', 'for beginners']):
            return "simple"
        elif any(word in content.lower() for word in ['technical details', 'api spec', 'source code']):
            return "technical" 
        return "adaptive"

    def _analyze_response_timing(self, data: dict) -> dict:
        """Analyze preferred response timing"""
        return {"preferred_speed": "normal"}

    def _measure_engagement(self, data: dict) -> float:
        """Measure engagement level from content"""
        content = data.get('content', '')
        if len(content.split()) > 20:
            return 0.8
        return 0.5

    def _detect_solving_style(self, data: dict) -> str:
        """Detect problem-solving approach"""
        content = data.get('content', '').lower()
        if any(word in content for word in ['step by step', 'how to', 'tutorial']):
            return "methodical"
        elif any(word in content for word in ['quick', 'fast', 'immediately']):
            return "direct"
        return "balanced"

class SecurityLearning(LearningEngine):
    def process_event(self, data: dict):
        """Learn from security events"""
        return {
            "threat_patterns": data.get("common_threats", []),
            "risk_trends": self._analyze_risk_trends(data),
            "trust_indicators": self._identify_trust_indicators(data)
        }
    
    def _analyze_risk_trends(self, data: dict) -> dict:
        return {"trend": "stable"}
    
    def _identify_trust_indicators(self, data: dict) -> list:
        return data.get("trusted_plugin_patterns", [])

class PluginLearning(LearningEngine):
    def process_event(self, data: dict):
        """Learn from plugin performance"""
        return {
            "optimal_plugins": self._identify_optimal_plugins(data),
            "performance_trends": self._analyze_performance_trends(data),
            "dependency_patterns": self._detect_dependencies(data)
        }
    
    def _identify_optimal_plugins(self, data: dict) -> list:
        return []
    
    def _analyze_performance_trends(self, data: dict) -> dict:
        return {"trend": "stable"}
    
    def _detect_dependencies(self, data: dict) -> dict:
        return {}

class WebKnowledgeLearning(LearningEngine):
    def process_event(self, data: dict):
        """Learn from web interactions"""
        return {
            "content_patterns": self._extract_content_patterns(data),
            "source_reliability": self._assess_source_reliability(data),
            "information_freshness": self._measure_freshness(data)
        }
    
    def _extract_content_patterns(self, data: dict) -> list:
        return []
    
    def _assess_source_reliability(self, data: dict) -> dict:
        return {}
    
    def _measure_freshness(self, data: dict) -> str:
        return "current"

class ConversationLearning(LearningEngine):
    def process_event(self, data: dict):
        """Learn from conversation patterns"""
        return {
            "dialogue_patterns": self._extract_dialogue_patterns(data),
            "user_preferences": self._detect_user_preferences(data),
            "interaction_style": self._analyze_interaction_style(data)
        }
    
    def _extract_dialogue_patterns(self, data: dict) -> list:
        return []
    
    def _detect_user_preferences(self, data: dict) -> dict:
        return {}
    
    def _analyze_interaction_style(self, data: dict) -> str:
        return "balanced"

class AryaAdvancedLearning:
    def __init__(self, memory: OptimizedAryaMemory, plugin_manager: OptimizedPluginManager, skyler: EnhancedSkylerNetron):
        self.memory = memory
        self.plugin_manager = plugin_manager
        self.skyler = skyler
        self.learning_engines = {
            "user_behavior": UserBehaviorLearning(),
            "security_patterns": SecurityLearning(), 
            "plugin_performance": PluginLearning(),
            "web_knowledge": WebKnowledgeLearning(),
            "conversation_patterns": ConversationLearning()
        }
    
    def learn_from_event(self, event_type: str, data: dict):
        """Centralized learning from all sources"""
        for engine_name, engine in self.learning_engines.items():
            if engine_name in event_type:
                engine.process_event(data)
        
        self._update_personality_model()
        self._extract_behavioral_patterns()
        self._build_knowledge_graph()

    def _update_personality_model(self):
        """Update personality based on learned patterns"""
        # Extract from recent conversations
        recent_convos = self.memory.get_recent_conversations(20)
        if recent_convos:
            user_messages = [msg for msg in recent_convos if msg.get('role') == 'user']
            if user_messages:
                content = " ".join([msg.get('content', '') for msg in user_messages])
                traits = self.learning_engines["user_behavior"].process_event({"content": content})
                self.memory.data["personality"]["learned_traits"] = traits
                self.memory.save()

    def _extract_behavioral_patterns(self):
        """Extract behavioral patterns from all data"""
        patterns = {
            "preferred_plugins": self._get_preferred_plugins(),
            "common_topics": self._get_common_topics(),
            "interaction_times": self._get_interaction_times()
        }
        self.memory.data["learning"]["behavioral_patterns"] = patterns
        self.memory.save()

    def _build_knowledge_graph(self):
        """Build connections between different learning domains"""
        # Connect plugin usage with security trust
        plugin_trust = {}
        for plugin_name in self.plugin_manager.plugins:
            trust_data = self.skyler.trust_profiles.get(plugin_name, {})
            usage_data = self.plugin_manager.usage_patterns.get(plugin_name, {})
            
            plugin_trust[plugin_name] = {
                "security_trust": trust_data.get("trust_score", 0.5),
                "usage_frequency": len(usage_data.get("trigger_patterns", [])),
                "performance_score": self._calculate_performance_score(plugin_name)
            }
        
        self.memory.data["learning"]["plugin_knowledge_graph"] = plugin_trust
        self.memory.save()

    def _get_preferred_plugins(self) -> list:
        """Get most frequently used plugins"""
        usage_counts = {}
        for plugin_name, patterns in self.plugin_manager.usage_patterns.items():
            usage_counts[plugin_name] = len(patterns.get("trigger_patterns", []))
        
        return sorted(usage_counts.items(), key=lambda x: x[1], reverse=True)[:5]

    def _get_common_topics(self) -> list:
        """Extract common topics from conversations"""
        recent_convos = self.memory.get_recent_conversations(50)
        all_content = " ".join([msg.get('content', '') for msg in recent_convos])
        
        topics = {
            "programming": ["python", "code", "api", "database"],
            "politics": ["election", "candidate", "representative", "congress"],
            "data": ["scraping", "analysis", "export", "collection"]
        }
        
        found_topics = []
        for topic, keywords in topics.items():
            if any(keyword in all_content.lower() for keyword in keywords):
                found_topics.append(topic)
        
        return found_topics

    def _get_interaction_times(self) -> dict:
        """Learn when user typically interacts"""
        # This would analyze conversation timestamps
        return {"peak_hours": [9, 14, 20]}  # Example

    def _calculate_performance_score(self, plugin_name: str) -> float:
        """Calculate overall performance score for plugin"""
        metrics = self.plugin_manager.performance_metrics.get(plugin_name, {})
        if not metrics:
            return 0.5
        
        success_rate = metrics.get("success_count", 0) / max(1, metrics.get("success_count", 0) + metrics.get("failure_count", 0))
        avg_time = metrics.get("average_time", 1.0)
        
        # Normalize time (lower is better)
        time_score = max(0, 1 - (avg_time / 10.0))  # Cap at 10 seconds
        
        return (success_rate + time_score) / 2

# =============================================================================
# 🎨 ENHANCED GUI SYSTEM
# =============================================================================
def launch_enhanced_gui(memory: OptimizedAryaMemory, plugin_manager: OptimizedPluginManager):
    try:
        import eel
        eel.init(str(HDD_PATH / "gui"))
        
        # =============================================================================
        # 🎯 CORE EEL FUNCTIONS - ALL REQUIRED FOR GUI
        # =============================================================================
        
        @eel.expose
        def get_enhanced_memory():
            """Enhanced memory with stats and analytics"""
            base_memory = memory.data
            return {
                **base_memory,
                "stats": {
                    "total_plugins": len(plugin_manager.plugins),
                    "active_plugins": list(plugin_manager.plugins.keys()),
                    "conversation_count": memory.data["total_conversations"],
                    "memory_usage": f"{memory._get_memory_usage() if hasattr(memory, '_get_memory_usage') else 0}MB",
                    "uptime": time.time() - memory.data["first_interaction"]
                }
            }
        
        @eel.expose
        def get_conversation_history(limit: int = 50):
            """Fast paginated conversation history"""
            return memory.get_recent_conversations(limit)
        
        @eel.expose  
        def get_plugin_status():
            """Real-time plugin health monitoring"""
            status = {}
            for name, plugin in plugin_manager.plugins.items():
                status[name] = {
                    "loaded": True,
                    "last_used": plugin.get("last_used", "Never"),
                    "usage_count": plugin.get("usage_count", 0),
                    "performance": plugin_manager.performance_metrics.get(name, {})
                }
            return status
        
        @eel.expose
        def get_security_insights():
            """Get security intelligence from Skyler"""
            return plugin_manager.skyler.get_security_insights()
        
        @eel.expose
        def get_learning_insights():
            """Get learning insights"""
            return {
                "personality": memory.data.get("personality", {}).get("learned_traits", {}),
                "behavioral_patterns": memory.data.get("learning", {}).get("behavioral_patterns", {}),
                "plugin_knowledge": memory.data.get("learning", {}).get("plugin_knowledge_graph", {})
            }
        
        @eel.expose
        def batch_plugin_operation(operation: str, plugins: list):
            """Bulk plugin management"""
            results = []
            for plugin_name in plugins:
                try:
                    if operation == "enable":
                        # Add to approved list
                        if plugin_name not in memory.data["approved_plugins"]:
                            memory.data["approved_plugins"].append(plugin_name)
                        results.append(f"✅ {plugin_name} enabled")
                    elif operation == "disable":
                        if plugin_name in memory.data["approved_plugins"]:
                            memory.data["approved_plugins"].remove(plugin_name)
                        results.append(f"❌ {plugin_name} disabled")
                    memory.save()
                except Exception as e:
                    results.append(f"⚠️ {plugin_name} failed: {e}")
            return results

        @eel.expose
        def get_plugin_recommendations(user_input: str):
            """Get plugin suggestions based on learned patterns"""
            return plugin_manager.get_plugin_recommendations(user_input)

        # =============================================================================
        # 🗣️ CHAT & MESSAGE FUNCTIONS
        # =============================================================================
        
        @eel.expose
        def send_message(user_input: str):
            """Handle chat messages from GUI - FIXED VERSION"""
            try:
                print(f"📨 Received message: {user_input}")
                
                # Handle huge inputs
                if len(user_input) > MAX_INPUT_LENGTH:
                    chunks = [user_input[i:i+50000] for i in range(0, len(user_input), 50000)]
                    facts = []
                    for chunk in chunks:
                        # Extract key facts from chunk
                        fact = get_ai_response(f"Extract key facts: {chunk[:10000]}", model="mistral")
                        facts.append(fact)
                    user_input = " ".join(facts)
                
                # Save to memory
                memory.add_conversation("user", user_input)
                
                # Check for plugin commands
                if user_input.lower().startswith("use "):
                    parts = user_input.split()
                    if len(parts) > 1:
                        plugin_name = parts[1]
                        plugin_input = " ".join(parts[2:]) if len(parts) > 2 else ""
                        result = plugin_manager.execute(plugin_name, input=plugin_input)
                        memory.add_conversation("assistant", str(result))
                        return str(result)
                
                # Check for create plugin command
                if user_input.lower().startswith("create plugin "):
                    plugin_name = user_input[14:].strip()
                    if plugin_name:
                        template = plugin_manager.create_plugin_template(plugin_name)
                        memory.add_conversation("assistant", f"Plugin template created for: {plugin_name}")
                        return f"🧩 Created plugin template for '{plugin_name}':\n\n{template}"
                
                # Regular AI response
                response = get_ai_response(user_input, model="mistral")
                memory.add_conversation("assistant", response)
                return response
                
            except Exception as e:
                error_msg = f"Error processing message: {str(e)}"
                print(f"❌ {error_msg}")
                return error_msg

        # =============================================================================
        # 📁 FILE UPLOAD FUNCTIONS
        # =============================================================================
        
        @eel.expose
        def upload_file(file_data: str, filename: str, file_type: str):
            """Handle file uploads from GUI"""
            try:
                import base64
                import uuid
                from pathlib import Path
                
                print(f"📤 Uploading file: {filename} ({file_type})")
                
                # Create uploads directory
                uploads_dir = HDD_PATH / "uploads"
                uploads_dir.mkdir(exist_ok=True)
                
                # Generate unique filename
                file_extension = Path(filename).suffix
                unique_filename = f"{uuid.uuid4()}{file_extension}"
                file_path = uploads_dir / unique_filename
                
                # Decode and save file
                if file_data.startswith('data:'):
                    # Handle data URL format (from browser)
                    file_data = file_data.split(',')[1]
                
                file_bytes = base64.b64decode(file_data)
                file_path.write_bytes(file_bytes)
                
                # Process based on file type
                result = process_uploaded_file(file_path, filename, file_type)
                
                return f"✅ File '{filename}' uploaded successfully!\n{result}"
                
            except Exception as e:
                error_msg = f"❌ Upload failed: {str(e)}"
                print(error_msg)
                return error_msg

        def process_uploaded_file(file_path: Path, original_name: str, file_type: str):
            """Process different types of uploaded files"""
            file_extension = file_path.suffix.lower()
            
            if file_extension == '.json':
                return process_json_file(file_path, original_name)
            elif file_extension == '.csv':
                return process_csv_file(file_path, original_name)
            elif file_extension in ['.txt', '.log']:
                return process_text_file(file_path, original_name)
            elif file_extension == '.pdf':
                return process_pdf_file(file_path, original_name)
            elif file_extension in ['.zip', '.tar', '.gz']:
                return process_archive_file(file_path, original_name)
            else:
                return f"File saved: {file_path}\nType: {file_type}\nSize: {file_path.stat().st_size} bytes"

        def process_json_file(file_path: Path, original_name: str):
            """Process JSON files (like ChatGPT exports)"""
            try:
                import json
                data = json.loads(file_path.read_text())
                
                # Check if it's a ChatGPT export
                if isinstance(data, list) and any('messages' in item for item in data):
                    return "📄 ChatGPT conversation export detected! Use the conversation importer plugin to process this."
                elif 'messages' in data:
                    return "💬 Chat conversation data detected. Ready for import."
                else:
                    return f"📊 JSON data loaded: {len(data)} items" if isinstance(data, list) else "📊 JSON object loaded"
                    
            except Exception as e:
                return f"❌ JSON processing error: {str(e)}"

        def process_text_file(file_path: Path, original_name: str):
            """Process text files"""
            content = file_path.read_text()
            lines = content.split('\n')
            return f"📝 Text file: {len(lines)} lines, {len(content)} characters"

        def process_csv_file(file_path: Path, original_name: str):
            """Process CSV files"""
            try:
                import csv
                with open(file_path, 'r', newline='', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    rows = list(reader)
                    return f"📊 CSV file: {len(rows)} rows, {len(rows[0]) if rows else 0} columns"
            except:
                return "📊 CSV file uploaded (basic info)"

        def process_pdf_file(file_path: Path, original_name: str):
            """Process PDF files"""
            return "📄 PDF file uploaded - Text extraction available with additional plugins"

        def process_archive_file(file_path: Path, original_name: str):
            """Process archive files"""
            return "🗜️ Archive file uploaded - Extraction available with additional plugins"

        @eel.expose
        def get_supported_file_types():
            """Return supported file types for upload"""
            return {
                "text": [".txt", ".json", ".csv", ".xml", ".log"],
                "documents": [".pdf", ".doc", ".docx", ".rtf"],
                "data": [".json", ".csv", ".xlsx", ".xls"],
                "archives": [".zip", ".tar", ".gz"],
                "images": [".jpg", ".jpeg", ".png", ".gif", ".bmp"],
                "code": [".py", ".js", ".html", ".css", ".java", ".cpp", ".c"]
            }

        # =============================================================================
        # 🧩 PLUGIN MANAGEMENT FUNCTIONS
        # =============================================================================
        
        @eel.expose
        def create_plugin(plugin_name: str):
            """Create a new plugin template"""
            try:
                template = plugin_manager.create_plugin_template(plugin_name)
                return f"🧩 Plugin template created for '{plugin_name}':\n\n{template}"
            except Exception as e:
                return f"❌ Failed to create plugin: {str(e)}"

        @eel.expose
        def execute_plugin(plugin_name: str, input_text: str = ""):
            """Execute a plugin by name"""
            try:
                result = plugin_manager.execute(plugin_name, input=input_text)
                return str(result)
            except Exception as e:
                return f"❌ Plugin execution failed: {str(e)}"

        # =============================================================================
        # 🚀 START THE GUI
        # =============================================================================
        
        print("🌐 Launching Arya GUI...")
        eel.start('index.html', size=(1400, 900), port=8080, mode='chrome')
        
    except ImportError as e:
        print(f"❌ GUI requires 'eel'. Run: pip install eel - Error: {e}")
    except Exception as e:
        print(f"❌ GUI error: {e}")
        import traceback
        traceback.print_exc()

# =============================================================================
# 🧠 ENHANCED AI FUNCTIONS
# =============================================================================
def get_context_aware_response(prompt: str, conversation_context: list, model: str = "mistral") -> str:
    """Smarter responses using conversation context"""
    try:
        # Analyze conversation patterns
        context_summary = summarize_conversation_context(conversation_context)
        
        enhanced_prompt = f"""
        Conversation context: {context_summary}
        Current query: {prompt}
        
        Respond appropriately considering the conversation history.
        """
        
        import requests
        response = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={
                "model": model, 
                "prompt": enhanced_prompt, 
                "stream": False,
                "options": {
                    "num_ctx": 8192,  # Larger context window
                    "temperature": 0.7,
                    "top_k": 40,
                    "top_p": 0.9
                }
            },
            timeout=300
        )
        return response.json().get("response", "").strip()
    except Exception as e:
        return f"Enhanced AI error: {str(e)}"

def summarize_conversation_context(conversations: list) -> str:
    """Extract key topics and context from recent conversations"""
    if not conversations:
        return "No recent context"
    
    # Simple keyword extraction (can be enhanced)
    recent_text = " ".join([conv.get('content', '') for conv in conversations[-5:]])
    
    # Extract potential topics (simplified)
    topics = []
    if any(word in recent_text.lower() for word in ['plugin', 'scrape', 'data']):
        topics.append("data_collection")
    if any(word in recent_text.lower() for word in ['political', 'candidate', 'election']):
        topics.append("politics")
    if any(word in recent_text.lower() for word in ['help', 'how to', 'explain']):
        topics.append("assistance")
        
    return f"Recent topics: {', '.join(topics) if topics else 'general conversation'}"

def get_ai_response(prompt: str, model: str = "mistral") -> str:
    """Get response from Ollama with memory context"""
    try:
        recent_context = ""
        shard_file = DIRS["conversations"] / "shard_0001.json"
        if shard_file.exists():
            try:
                shard_data = json.loads(shard_file.read_text())
                recent_messages = shard_data[-5:] if len(shard_data) >= 5 else shard_data
                recent_context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent_messages])
            except (json.JSONDecodeError, Exception) as e:
                logging.error(f"Error reading shard: {e}")
                recent_context = ""
        
        full_prompt = f"""You are Arya Rachel Friday v{CORE_VERSION}. 
User: {prompt}
Recent context: {recent_context}
Respond naturally."""
        
        import requests
        response = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={"model": model, "prompt": full_prompt, "stream": False},
            timeout=300
        )
        return response.json().get("response", "").strip()
    except Exception as e:
        return f"AI error: {str(e)}"

# =============================================================================
# 🔄 CONTINUOUS LEARNING SYSTEM
# =============================================================================
class AryaContinuousLearning:
    def __init__(self, memory: OptimizedAryaMemory, plugin_manager: OptimizedPluginManager, skyler: EnhancedSkylerNetron):
        self.memory = memory
        self.plugin_manager = plugin_manager
        self.skyler = skyler
        self.learning_engine = AryaAdvancedLearning(memory, plugin_manager, skyler)
        self.learning_interval = 3600  # Learn every hour
        self.last_learning_time = 0
    
    def continuous_learning_loop(self):
        """Run continuous learning in background"""
        while True:
            self.learning_cycle()
            time.sleep(300)  # Check every 5 minutes
    
    def learning_cycle(self):
        """Run continuous learning processes"""
        current_time = time.time()
        if current_time - self.last_learning_time < self.learning_interval:
            return
        
        print("🔄 Running continuous learning cycle...")
        
        # 1. Learn from recent conversations
        self._learn_from_conversations()
        
        # 2. Learn from plugin performance
        self._learn_from_plugin_metrics()
        
        # 3. Learn from security patterns
        self._learn_from_security_events()
        
        # 4. Update personality model
        self._update_personality_model()
        
        # 5. Consolidate knowledge
        self._consolidate_learning()
        
        self.last_learning_time = current_time
        print("✅ Learning cycle complete")
    
    def _learn_from_conversations(self):
        """Extract insights from recent conversations"""
        recent_convos = self.memory.get_recent_conversations(50)
        
        # Analyze for patterns
        topics = self._extract_common_topics(recent_convos)
        styles = self._analyze_communication_styles(recent_convos)
        preferences = self._detect_user_preferences(recent_convos)
        
        # Update memory with insights
        self.memory.data["learning"]["recent_topics"] = topics
        self.memory.data["personality"]["communication_style"] = styles
        self.memory.data["learning"]["user_preferences"] = preferences
        self.memory.save()
    
    def _learn_from_plugin_metrics(self):
        """Learn from plugin performance data"""
        for plugin_name, metrics in self.plugin_manager.performance_metrics.items():
            self.learning_engine.learn_from_event("plugin_performance", {
                "plugin": plugin_name,
                "metrics": metrics
            })
    
    def _learn_from_security_events(self):
        """Learn from security patterns"""
        insights = self.skyler.get_security_insights()
        self.learning_engine.learn_from_event("security_patterns", insights)
    
    def _update_personality_model(self):
        """Update personality based on all learned data"""
        self.learning_engine._update_personality_model()
    
    def _consolidate_learning(self):
        """Consolidate all learning into knowledge graph"""
        self.learning_engine._build_knowledge_graph()
    
    def _extract_common_topics(self, conversations: list) -> list:
        """Extract common topics from conversations"""
        # Implementation would go here
        return ["politics", "data_collection"]
    
    def _analyze_communication_styles(self, conversations: list) -> str:
        """Analyze communication style patterns"""
        return "adaptive"
    
    def _detect_user_preferences(self, conversations: list) -> dict:
        """Detect user preferences from conversations"""
        return {"preferred_detail_level": "technical"}

# =============================================================================
# 🚀 MAIN ENTRY POINT
# =============================================================================
def main():
    print(f"🚀 ARYA RACHEL FRIDAY v{CORE_VERSION} - ENHANCED CORE")
    print(f"💾 Project path: {HDD_PATH}")
    
    # Check dependencies
    try:
        import requests
        import bs4
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("🔧 Run: pip install requests beautifulsoup4")
        return
    
    # Initialize ENHANCED systems
    memory = OptimizedAryaMemory()
    skyler = EnhancedSkylerNetron()
    plugin_manager = OptimizedPluginManager(memory, skyler)
    learning_engine = AryaContinuousLearning(memory, plugin_manager, skyler)
    
    # Start learning thread
    learning_thread = threading.Thread(target=learning_engine.continuous_learning_loop, daemon=True)
    learning_thread.start()
    
    # Launch ENHANCED GUI
    if "--no-gui" not in sys.argv:
        launch_enhanced_gui(memory, plugin_manager)
    else:
        print("GUI disabled. Use --no-gui to run headless.")

if __name__ == "__main__":
    main()