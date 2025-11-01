import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class BasketFormatApriori:
    def __init__(self, dataset_file='groceries - groceries.csv'):
        """
        Apriori analysis for basket format CSV files
        
        Args:
            dataset_file (str): Path to the basket format CSV file
        """
        self.dataset_file = dataset_file
        self.df_raw = None
        self.transactions = []
        self.basket_matrix = None
        self.frequent_itemsets = None
        self.rules = None
        self.item_stats = None
        
    def load_and_process_data(self):
        """Load and process the basket format data"""
        print("=" * 70)
        print("APRIORI ANALYSIS - BASKET FORMAT")
        print("=" * 70)
        
        try:
            self.df_raw = pd.read_csv(self.dataset_file)
            print(f"Successfully loaded {self.dataset_file}")
        except FileNotFoundError:
            print(f"Error: {self.dataset_file} not found!")
            return False
            
        print(f"Raw Data Overview:")
        print(f"   - Total transactions: {len(self.df_raw):,}")
        print(f"   - Maximum items per transaction: {self.df_raw.shape[1] - 1}")
        
        # Process transactions
        self.transactions = []
        for idx, row in self.df_raw.iterrows():
            # Get all items from the row (excluding the first column which is item count)
            items = []
            for col in self.df_raw.columns[1:]:  # Skip first column
                item = row[col]
                if pd.notna(item) and item != '' and str(item).strip() != '':
                    items.append(str(item).strip())
            
            if items:  # Only add non-empty transactions
                self.transactions.append(items)
        
        print(f"   - Valid transactions processed: {len(self.transactions):,}")
        
        return True
    
    def analyze_transaction_patterns(self):
        """Analyze transaction patterns and item frequencies"""
        print(f"\nTransaction Analysis:")
        
        # Calculate transaction sizes
        transaction_sizes = [len(transaction) for transaction in self.transactions]
        
        print(f"   - Average basket size: {np.mean(transaction_sizes):.2f} items")
        print(f"   - Median basket size: {np.median(transaction_sizes):.0f} items")
        print(f"   - Min basket size: {min(transaction_sizes)} items")
        print(f"   - Max basket size: {max(transaction_sizes)} items")
        
        # Basket size distribution
        size_counts = Counter(transaction_sizes)
        print(f"\nBasket Size Distribution:")
        for size in sorted(size_counts.keys())[:10]:  # Show top 10
            count = size_counts[size]
            percentage = (count / len(self.transactions)) * 100
            print(f"   - {size} items: {count:,} transactions ({percentage:.1f}%)")
        
        # Item frequency analysis
        all_items = []
        for transaction in self.transactions:
            all_items.extend(transaction)
        
        item_counts = Counter(all_items)
        self.item_stats = pd.DataFrame(list(item_counts.items()), 
                                     columns=['Item', 'Frequency'])
        self.item_stats['Support'] = self.item_stats['Frequency'] / len(self.transactions)
        self.item_stats = self.item_stats.sort_values('Frequency', ascending=False).reset_index(drop=True)
        
        print(f"\nTop 15 Most Popular Items:")
        for i, row in self.item_stats.head(15).iterrows():
            print(f"   {i+1:2d}. {row['Item']:<25} ({row['Frequency']:,} times, {row['Support']*100:.1f}%)")
        
        print(f"\nItem Statistics:")
        print(f"   - Total unique items: {len(self.item_stats):,}")
        print(f"   - Items appearing in >1% of transactions: {(self.item_stats['Support'] > 0.01).sum()}")
        print(f"   - Items appearing in >5% of transactions: {(self.item_stats['Support'] > 0.05).sum()}")
        
        return True
    
    def create_basket_matrix(self, min_support=0.01):
        """
        Create one-hot encoded basket matrix
        
        Args:
            min_support (float): Minimum support to include items
        """
        print(f"\nCreating Basket Matrix:")
        
        # Filter items by minimum support
        frequent_items = self.item_stats[self.item_stats['Support'] >= min_support]['Item'].tolist()
        
        print(f"   - Items before filtering: {len(self.item_stats)}")
        print(f"   - Items after filtering (min support {min_support}): {len(frequent_items)}")
        
        # Create binary matrix
        matrix_data = []
        for transaction in self.transactions:
            row = {}
            for item in frequent_items:
                row[item] = 1 if item in transaction else 0
            matrix_data.append(row)
        
        self.basket_matrix = pd.DataFrame(matrix_data)
        
        # Ensure all columns are present (in case some items don't appear in any transaction)
        for item in frequent_items:
            if item not in self.basket_matrix.columns:
                self.basket_matrix[item] = 0
        
        print(f"   - Final matrix shape: {self.basket_matrix.shape}")
        
        # Calculate sparsity
        total_cells = self.basket_matrix.shape[0] * self.basket_matrix.shape[1]
        filled_cells = self.basket_matrix.sum().sum()
        sparsity = (1 - filled_cells / total_cells) * 100
        print(f"   - Matrix sparsity: {sparsity:.1f}%")
        
        return True
    
    def find_frequent_itemsets(self, min_support=0.01):
        """
        Find frequent itemsets using Apriori algorithm
        
        Args:
            min_support (float): Minimum support threshold
        """
        print(f"\nRunning Apriori Algorithm:")
        print(f"   - Minimum support threshold: {min_support}")
        
        # Run Apriori
        self.frequent_itemsets = apriori(
            self.basket_matrix, 
            min_support=min_support, 
            use_colnames=True,
            verbose=1
        )
        
        if len(self.frequent_itemsets) == 0:
            print("   WARNING: No frequent itemsets found! Try lowering the support threshold.")
            return False
            
        print(f"   - Found {len(self.frequent_itemsets)} frequent itemsets")
        
        # Show itemset size distribution
        itemset_sizes = self.frequent_itemsets['itemsets'].apply(len)
        print(f"\nFrequent Itemset Distribution:")
        for size in sorted(itemset_sizes.unique()):
            count = (itemset_sizes == size).sum()
            print(f"   - Size {size}: {count} itemsets")
        
        # Show top frequent itemsets by size
        if len(self.frequent_itemsets) > 0:
            print(f"\nTop Frequent Itemsets:")
            
            # Single items (size 1)
            size_1 = self.frequent_itemsets[itemset_sizes == 1].nlargest(5, 'support')
            if len(size_1) > 0:
                print(f"   Single Items:")
                for _, row in size_1.iterrows():
                    item = list(row['itemsets'])[0]
                    print(f"      - {item:<30} (support: {row['support']:.3f})")
            
            # Item pairs (size 2)
            size_2 = self.frequent_itemsets[itemset_sizes == 2].nlargest(5, 'support')
            if len(size_2) > 0:
                print(f"   Item Pairs:")
                for _, row in size_2.iterrows():
                    items = ', '.join(list(row['itemsets']))
                    print(f"      - {items:<40} (support: {row['support']:.3f})")
        
        return True
    
    def generate_association_rules(self, min_confidence=0.1, min_lift=1.0):
        """
        Generate association rules
        
        Args:
            min_confidence (float): Minimum confidence threshold
            min_lift (float): Minimum lift threshold
        """
        print(f"\nGenerating Association Rules:")
        print(f"   - Minimum confidence: {min_confidence}")
        print(f"   - Minimum lift: {min_lift}")
        
        # Generate rules
        self.rules = association_rules(
            self.frequent_itemsets, 
            metric="confidence", 
            min_threshold=min_confidence
        )
        
        if len(self.rules) == 0:
            print("   WARNING: No rules found! Try lowering the confidence threshold.")
            return False
        
        # Filter by lift
        self.rules = self.rules[self.rules['lift'] >= min_lift]
        
        if len(self.rules) == 0:
            print(f"   WARNING: No rules found with lift >= {min_lift}!")
            return False
            
        # Sort by lift and confidence
        self.rules = self.rules.sort_values(['lift', 'confidence'], ascending=False).reset_index(drop=True)
        
        print(f"   - Generated {len(self.rules)} association rules")
        
        # Show rule statistics
        print(f"\nRule Statistics:")
        print(f"   - Confidence range: {self.rules['confidence'].min():.3f} to {self.rules['confidence'].max():.3f}")
        print(f"   - Lift range: {self.rules['lift'].min():.3f} to {self.rules['lift'].max():.3f}")
        print(f"   - Average confidence: {self.rules['confidence'].mean():.3f}")
        print(f"   - Average lift: {self.rules['lift'].mean():.3f}")
        
        return True
    
    def display_top_rules(self, top_n=15):
        """Display the top association rules"""
        print(f"\nTop {min(top_n, len(self.rules))} Association Rules:")
        print("=" * 100)
        
        for i, row in self.rules.head(top_n).iterrows():
            antecedent = ', '.join(list(row['antecedents']))
            consequent = ', '.join(list(row['consequents']))
            
            print(f"{i+1:2d}. {antecedent:<35} -> {consequent:<35}")
            print(f"    Support: {row['support']:.4f} | Confidence: {row['confidence']:.3f} | Lift: {row['lift']:.3f}")
            
            # Business interpretation
            confidence_pct = row['confidence'] * 100
            if row['lift'] > 1.2:
                strength = "Strong"
            elif row['lift'] > 1.1:
                strength = "Moderate"
            else:
                strength = "Weak"
                
            print(f"    INTERPRETATION: {confidence_pct:.1f}% of customers buying '{antecedent}' also buy '{consequent}'")
            print(f"    STRENGTH: {strength} association ({row['lift']:.2f}x more likely than random)")
            print("-" * 100)
    
    def create_visualizations(self):
        """Create comprehensive visualizations"""
        if self.rules is None or len(self.rules) == 0:
            print("No rules to visualize!")
            return
            
        # Set up the plotting style
        plt.style.use('default')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Market Basket Analysis - Basket Format Data', fontsize=16, fontweight='bold')
        
        # 1. Support vs Confidence
        ax1 = axes[0, 0]
        scatter = ax1.scatter(self.rules['support'], self.rules['confidence'], 
                             c=self.rules['lift'], s=80, alpha=0.7, cmap='viridis')
        ax1.set_xlabel('Support')
        ax1.set_ylabel('Confidence')
        ax1.set_title('Support vs Confidence (Color = Lift)')
        plt.colorbar(scatter, ax=ax1, label='Lift')
        ax1.grid(True, alpha=0.3)
        
        # 2. Lift distribution
        ax2 = axes[0, 1]
        ax2.hist(self.rules['lift'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.axvline(1.0, color='red', linestyle='--', label='Random (Lift=1)')
        ax2.set_xlabel('Lift')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Lift Values')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Top items frequency
        ax3 = axes[0, 2]
        top_items = self.item_stats.head(10)
        ax3.barh(range(len(top_items)), top_items['Frequency'], alpha=0.7, color='lightcoral')
        ax3.set_yticks(range(len(top_items)))
        ax3.set_yticklabels([item[:20] + '...' if len(item) > 20 else item for item in top_items['Item']])
        ax3.set_xlabel('Frequency')
        ax3.set_title('Top 10 Most Frequent Items')
        ax3.grid(True, alpha=0.3)
        
        # 4. Confidence vs Lift
        ax4 = axes[1, 0]
        scatter2 = ax4.scatter(self.rules['confidence'], self.rules['lift'], 
                              c=self.rules['support'], s=80, alpha=0.7, cmap='plasma')
        ax4.set_xlabel('Confidence')
        ax4.set_ylabel('Lift')
        ax4.set_title('Confidence vs Lift (Color = Support)')
        ax4.axhline(1.0, color='red', linestyle='--', alpha=0.5, label='Random (Lift=1)')
        plt.colorbar(scatter2, ax=ax4, label='Support')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Transaction size distribution
        ax5 = axes[1, 1]
        transaction_sizes = [len(transaction) for transaction in self.transactions]
        size_counts = Counter(transaction_sizes)
        sizes = sorted(size_counts.keys())[:15]  # Top 15 sizes
        counts = [size_counts[size] for size in sizes]
        
        ax5.bar(sizes, counts, alpha=0.7, color='lightgreen', edgecolor='black')
        ax5.set_xlabel('Basket Size (Number of Items)')
        ax5.set_ylabel('Number of Transactions')
        ax5.set_title('Transaction Size Distribution')
        ax5.grid(True, alpha=0.3)
        
        # 6. Top rules by lift
        ax6 = axes[1, 2]
        top_rules = self.rules.head(8)
        rule_labels = []
        for _, row in top_rules.iterrows():
            ant = ', '.join(list(row['antecedents']))
            con = ', '.join(list(row['consequents']))
            label = f"{ant} → {con}"
            rule_labels.append(label[:25] + '...' if len(label) > 25 else label)
        
        y_pos = np.arange(len(rule_labels))
        ax6.barh(y_pos, top_rules['lift'], alpha=0.7, color='gold')
        ax6.set_yticks(y_pos)
        ax6.set_yticklabels(rule_labels, fontsize=8)
        ax6.set_xlabel('Lift')
        ax6.set_title('Top Rules by Lift')
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('basket_format_apriori_analysis.png', dpi=300, bbox_inches='tight')
        print(f"\nVisualizations saved as 'basket_format_apriori_analysis.png'")
        plt.show()
    
    def run_complete_analysis(self, min_support=0.01, min_confidence=0.1, min_lift=1.0):
        """
        Run the complete Apriori analysis pipeline
        
        Args:
            min_support (float): Minimum support for frequent itemsets
            min_confidence (float): Minimum confidence for association rules
            min_lift (float): Minimum lift for association rules
        """
        # Load and process data
        if not self.load_and_process_data():
            return False
            
        # Analyze transaction patterns
        if not self.analyze_transaction_patterns():
            return False
            
        # Create basket matrix
        if not self.create_basket_matrix(min_support):
            return False
            
        # Find frequent itemsets
        if not self.find_frequent_itemsets(min_support):
            return False
            
        # Generate association rules
        if not self.generate_association_rules(min_confidence, min_lift):
            return False
            
        # Display results
        self.display_top_rules()
        
        # Create visualizations
        self.create_visualizations()
        
        print(f"\nComplete analysis finished successfully!")
        print(f"Summary:")
        print(f"   - Transactions analyzed: {len(self.transactions):,}")
        print(f"   - Unique items: {len(self.item_stats):,}")
        print(f"   - Frequent itemsets found: {len(self.frequent_itemsets):,}")
        print(f"   - Association rules generated: {len(self.rules):,}")
        
        return True

# Main execution
if __name__ == "__main__":

    analyzer = BasketFormatApriori('groceries - groceries.csv')
    

    analyzer.run_complete_analysis(
        min_support=0.01,       # 1% minimum support
        min_confidence=0.25,    # 25% minimum confidence
        min_lift=1.1           
    )