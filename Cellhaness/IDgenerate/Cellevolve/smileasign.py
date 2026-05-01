import os
import sys
import json
import time
import random
import schedule
from datetime import datetime
from threading import Thread

class SmileDispatcher:
    """SMILE自动派发器"""
    
    def __init__(self, config_path='cellselfevolve.json'):
        self.config_path = config_path
        self.config = self._load_config()
        self.current_smile = None
        self.last_dispatch_time = None
        self.is_running = False
        self.thread = None
        
    def _load_config(self):
        """加载配置文件"""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                return json.load(f)
        else:
            return self._create_default_config()
    
    def _create_default_config(self):
        """创建默认配置"""
        default_config = {
            "auto_dispatch": {
                "enabled": True,
                "interval_hours": 6,
                "last_dispatch": None
            },
            "drug_library": {
                "anti_inflammatory": [
                    {"smile": "CC(=O)OC1=CC=CC=C1C(=O)O", "name": "Aspirin", "category": "抗炎"},
                    {"smile": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O", "name": "Ibuprofen", "category": "抗炎"},
                    {"smile": "C1=CC=CC=C1C(=O)O", "name": "Benzoic acid", "category": "抗炎"},
                    {"smile": "C1CCCCC1C(=O)O", "name": "Cyclohexanecarboxylic acid", "category": "抗炎"}
                ],
                "immunomodulator": [
                    {"smile": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "name": "Caffeine", "category": "免疫调节"},
                    {"smile": "CCOC(=O)C(C)CC1=CC=C(C=C1)O", "name": "Phenoxyacetic acid", "category": "免疫调节"}
                ],
                "cytokine_inhibitor": [
                    {"smile": "CN(C)CC1=CC=C(C=C1)O", "name": "Phenol derivative", "category": "细胞因子抑制"},
                    {"smile": "C1=CC=C(C=C1)CC(=O)O", "name": "Phenylacetic acid", "category": "细胞因子抑制"}
                ],
                "checkpoint_inhibitor": [
                    {"smile": "C1CCN(CC1)C(=O)C1=CC=CC=C1", "name": "Carbanilide", "category": "免疫检查点"},
                    {"smile": "CN(C)C(=O)C1=CC=C(C=C1)NC(=O)N", "name": "Urea derivative", "category": "免疫检查点"}
                ],
                "custom": []
            },
            "dispatch_rules": {
                "strategy": "random",  # random, round_robin, category_based
                "preferred_category": None,
                "avoid_repeat": True,
                "max_consecutive_same_category": 3
            },
            "notification": {
                "enabled": True,
                "log_file": "smile_dispatch.log"
            }
        }
        
        with open(self.config_path, 'w') as f:
            json.dump(default_config, f, indent=2)
        
        return default_config
    
    def save_config(self):
        """保存配置"""
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def get_all_drugs(self):
        """获取所有药物"""
        all_drugs = []
        for category, drugs in self.config['drug_library'].items():
            all_drugs.extend(drugs)
        return all_drugs
    
    def get_drugs_by_category(self, category):
        """按类别获取药物"""
        if category in self.config['drug_library']:
            return self.config['drug_library'][category]
        return []
    
    def add_custom_drug(self, smile, name, category='custom'):
        """添加自定义药物"""
        drug = {"smile": smile, "name": name, "category": category}
        
        if category not in self.config['drug_library']:
            self.config['drug_library'][category] = []
        
        # 检查是否已存在
        existing_smiles = [d['smile'] for d in self.config['drug_library'][category]]
        if smile not in existing_smiles:
            self.config['drug_library'][category].append(drug)
            self.save_config()
            self._log(f"Added custom drug: {name} ({smile})")
            return True
        return False
    
    def remove_drug(self, smile):
        """移除药物"""
        for category, drugs in self.config['drug_library'].items():
            for i, drug in enumerate(drugs):
                if drug['smile'] == smile:
                    removed = self.config['drug_library'][category].pop(i)
                    self.save_config()
                    self._log(f"Removed drug: {removed['name']} ({smile})")
                    return True
        return False
    
    def select_smile(self):
        """根据策略选择smile"""
        strategy = self.config['dispatch_rules']['strategy']
        avoid_repeat = self.config['dispatch_rules']['avoid_repeat']
        preferred_category = self.config['dispatch_rules']['preferred_category']
        
        all_drugs = self.get_all_drugs()
        
        if not all_drugs:
            return None
        
        if preferred_category and preferred_category in self.config['drug_library']:
            drugs = self.get_drugs_by_category(preferred_category)
        else:
            drugs = all_drugs
        
        if avoid_repeat and self.current_smile:
            drugs = [d for d in drugs if d['smile'] != self.current_smile]
        
        if not drugs:
            drugs = all_drugs
        
        if strategy == 'random':
            selected = random.choice(drugs)
        elif strategy == 'round_robin':
            # 简单的轮询实现
            current_idx = next((i for i, d in enumerate(all_drugs) if d['smile'] == self.current_smile), -1)
            selected = all_drugs[(current_idx + 1) % len(all_drugs)]
        else:
            selected = random.choice(drugs)
        
        return selected
    
    def dispatch(self):
        """执行派发"""
        drug = self.select_smile()
        
        if drug:
            self.current_smile = drug['smile']
            self.last_dispatch_time = datetime.now().isoformat()
            self.config['auto_dispatch']['last_dispatch'] = self.last_dispatch_time
            self.save_config()
            
            message = f"[{datetime.now()}] Dispatched SMILE: {drug['name']} ({drug['smile']})"
            self._log(message)
            print(message)
            
            # 通知cellselfevolve
            self._notify_cellselfevolve(drug)
            
            return drug
        else:
            self._log("No drugs available for dispatch")
            return None
    
    def _notify_cellselfevolve(self, drug):
        """通知cellselfevolve新smile"""
        # 创建派发记录文件
        dispatch_info = {
            'smile': drug['smile'],
            'name': drug['name'],
            'category': drug['category'],
            'dispatch_time': self.last_dispatch_time,
            'status': 'pending'
        }
        
        with open('/tmp/current_smile.json', 'w') as f:
            json.dump(dispatch_info, f, indent=2)
    
    def _log(self, message):
        """记录日志"""
        if self.config['notification']['enabled']:
            log_file = self.config['notification']['log_file']
            with open(log_file, 'a') as f:
                f.write(f"{message}\n")
    
    def set_auto_dispatch(self, enabled):
        """设置自动派发开关"""
        self.config['auto_dispatch']['enabled'] = enabled
        self.save_config()
        
        if enabled:
            self._log("Auto dispatch enabled")
            print("✓ Auto dispatch enabled")
        else:
            self._log("Auto dispatch disabled")
            print("✓ Auto dispatch disabled")
    
    def set_interval(self, hours):
        """设置派发间隔（小时）"""
        self.config['auto_dispatch']['interval_hours'] = hours
        self.save_config()
        self._log(f"Dispatch interval set to {hours} hours")
        print(f"✓ Dispatch interval set to {hours} hours")
    
    def set_strategy(self, strategy):
        """设置派发策略"""
        valid_strategies = ['random', 'round_robin', 'category_based']
        if strategy in valid_strategies:
            self.config['dispatch_rules']['strategy'] = strategy
            self.save_config()
            self._log(f"Dispatch strategy set to: {strategy}")
            print(f"✓ Dispatch strategy set to: {strategy}")
        else:
            print(f"✗ Invalid strategy. Valid options: {valid_strategies}")
    
    def set_preferred_category(self, category):
        """设置偏好类别"""
        if category in self.config['drug_library'] or category is None:
            self.config['dispatch_rules']['preferred_category'] = category
            self.save_config()
            self._log(f"Preferred category set to: {category}")
            print(f"✓ Preferred category set to: {category}")
        else:
            print(f"✗ Invalid category. Available: {list(self.config['drug_library'].keys())}")
    
    def _run_scheduler(self):
        """运行调度器"""
        while self.is_running:
            schedule.run_pending()
            time.sleep(1)
    
    def start(self):
        """启动自动派发服务"""
        if not self.config['auto_dispatch']['enabled']:
            print("Auto dispatch is disabled. Enable it first.")
            return
        
        interval = self.config['auto_dispatch']['interval_hours']
        
        print(f"Starting SMILE dispatcher...")
        print(f"  Interval: {interval} hours")
        print(f"  Strategy: {self.config['dispatch_rules']['strategy']}")
        print(f"  Preferred category: {self.config['dispatch_rules']['preferred_category']}")
        
        # 立即执行一次派发
        self.dispatch()
        
        # 设置定时任务
        schedule.every(interval).hours.do(self.dispatch)
        
        # 在后台线程运行
        self.is_running = True
        self.thread = Thread(target=self._run_scheduler, daemon=True)
        self.thread.start()
        
        print("✓ SMILE dispatcher started")
    
    def stop(self):
        """停止自动派发服务"""
        self.is_running = False
        if self.thread:
            self.thread.join()
        schedule.clear()
        self._log("SMILE dispatcher stopped")
        print("✓ SMILE dispatcher stopped")
    
    def status(self):
        """显示状态"""
        print("\n" + "=" * 60)
        print("SMILE DISPATCHER STATUS")
        print("=" * 60)
        print(f"Auto dispatch: {'Enabled' if self.config['auto_dispatch']['enabled'] else 'Disabled'}")
        print(f"Interval: {self.config['auto_dispatch']['interval_hours']} hours")
        print(f"Current strategy: {self.config['dispatch_rules']['strategy']}")
        print(f"Preferred category: {self.config['dispatch_rules']['preferred_category']}")
        print(f"Current SMILE: {self.current_smile}")
        print(f"Last dispatch: {self.last_dispatch_time}")
        print(f"\nDrug library categories:")
        for category, drugs in self.config['drug_library'].items():
            print(f"  {category}: {len(drugs)} drugs")
        print("=" * 60)
    
    def list_drugs(self, category=None):
        """列出药物"""
        if category:
            drugs = self.get_drugs_by_category(category)
            print(f"\nDrugs in category '{category}':")
        else:
            drugs = self.get_all_drugs()
            print("\nAll drugs:")
        
        for i, drug in enumerate(drugs, 1):
            print(f"  {i}. {drug['name']} - {drug['smile']} [{drug['category']}]")

def main():
    dispatcher = SmileDispatcher('cellselfevolve.json')
    
    if len(sys.argv) < 2:
        print("Usage: python smileasign.py <command> [options]")
        print("\nCommands:")
        print("  start          - Start the dispatcher")
        print("  stop           - Stop the dispatcher")
        print("  status         - Show status")
        print("  dispatch       - Manual dispatch")
        print("  enable         - Enable auto dispatch")
        print("  disable        - Disable auto dispatch")
        print("  list           - List all drugs")
        print("  add <smile> <name> [category] - Add custom drug")
        print("  remove <smile> - Remove drug")
        print("  set_interval <hours> - Set dispatch interval")
        print("  set_strategy <strategy> - Set strategy (random/round_robin)")
        print("  set_category <category> - Set preferred category")
        return
    
    command = sys.argv[1]
    
    if command == 'start':
        dispatcher.start()
        # 保持运行
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            dispatcher.stop()
    
    elif command == 'stop':
        dispatcher.stop()
    
    elif command == 'status':
        dispatcher.status()
    
    elif command == 'dispatch':
        drug = dispatcher.dispatch()
        if drug:
            print(f"\nDispatched: {drug['name']} ({drug['smile']})")
    
    elif command == 'enable':
        dispatcher.set_auto_dispatch(True)
    
    elif command == 'disable':
        dispatcher.set_auto_dispatch(False)
    
    elif command == 'list':
        category = sys.argv[2] if len(sys.argv) > 2 else None
        dispatcher.list_drugs(category)
    
    elif command == 'add':
        if len(sys.argv) >= 4:
            smile = sys.argv[2]
            name = sys.argv[3]
            category = sys.argv[4] if len(sys.argv) > 4 else 'custom'
            success = dispatcher.add_custom_drug(smile, name, category)
            if success:
                print(f"✓ Added drug: {name}")
            else:
                print("✗ Drug already exists")
        else:
            print("Usage: python smileasign.py add <smile> <name> [category]")
    
    elif command == 'remove':
        if len(sys.argv) >= 3:
            smile = sys.argv[2]
            success = dispatcher.remove_drug(smile)
            if success:
                print(f"✓ Removed drug")
            else:
                print("✗ Drug not found")
        else:
            print("Usage: python smileasign.py remove <smile>")
    
    elif command == 'set_interval':
        if len(sys.argv) >= 3:
            hours = int(sys.argv[2])
            dispatcher.set_interval(hours)
        else:
            print("Usage: python smileasign.py set_interval <hours>")
    
    elif command == 'set_strategy':
        if len(sys.argv) >= 3:
            strategy = sys.argv[2]
            dispatcher.set_strategy(strategy)
        else:
            print("Usage: python smileasign.py set_strategy <strategy>")
    
    elif command == 'set_category':
        if len(sys.argv) >= 3:
            category = sys.argv[2]
            dispatcher.set_preferred_category(category)
        else:
            print("Usage: python smileasign.py set_category <category>")
    
    else:
        print(f"Unknown command: {command}")

if __name__ == '__main__':
    main()
