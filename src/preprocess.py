import json
import pandas as pd
import re
from pathlib import Path
import csv
import os

# Agora usamos um diretório com múltiplos arquivos JSON
JSON_DIR = Path('data/02-jsonified/')

# Função para limpar links no formato Markdown
def clean_text(text):
    # Remove links no estilo [Texto](link)
    cleaned = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
    
    # Substitui quebras de linha por espaço
    cleaned = cleaned.replace('\n', ' ').replace('\r', ' ')

    # Remove espaços duplos ou triplos que podem surgir
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()

    return cleaned

def load_texts_from_directory(json_dir):
    examples = []
    json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
    
    if not json_files:
        raise ValueError(f"Nenhum arquivo JSON encontrado em {json_dir}")
    
    for json_file in json_files:
        file_path = json_dir / json_file
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # Verifica se é uma lista de itens
                if not isinstance(data, list):
                    print(f"⚠️ Atenção: {json_file} não contém uma lista. Ignorando arquivo.")
                    continue
                
                for item in data:
                    # Verifica se tem os campos necessários
                    if not all(key in item for key in ['text', 'interest']):
                        print(f"⚠️ Atenção: Item malformado em {json_file}. Campos obrigatórios ausentes.")
                        continue
                    
                    # Processa apenas se o texto for válido
                    if isinstance(item['text'], str) and item['text'].strip():
                        cleaned_text = clean_text(item['text'].strip())
                        
                        # Determina o label baseado no campo 'interest'
                        label = 1 if item['interest'].lower() == 'sim' else 0
                        
                        examples.append({
                            'text': cleaned_text,
                            'label': label,
                            'source_file': json_file  # Opcional: guarda origem do dado
                        })
                    else:
                        print(f"⚠️ Atenção: Texto vazio/inválido em {json_file}")
                        
        except Exception as e:
            print(f"❌ Erro ao processar {json_file}: {str(e)}")
    
    return examples

def main():
    try:
        all_data = load_texts_from_directory(JSON_DIR)
        
        if not all_data:
            raise ValueError("Nenhum dado válido encontrado nos arquivos JSON")
        
        df = pd.DataFrame(all_data)
        
        # Embaralha os dados
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Remove a coluna source_file se não for necessária
        df = df[['text', 'label']]
        
        output_dir = Path('data/03-ready_4_training/')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(
            output_dir / 'classifier-dataset.csv',
            index=False,
            encoding='utf-8',
            quoting=csv.QUOTE_ALL,
            doublequote=True
        )
        
        # Estatísticas úteis
        sim_count = df[df['label'] == 1].shape[0]
        nao_count = df[df['label'] == 0].shape[0]
        
        print(f"✅ Dataset salvo em {output_dir / 'classifier-dataset.csv'}")
        print(f"📊 Estatísticas:")
        print(f"   - Total de exemplos: {len(df)}")
        print(f"   - Classe 'Sim': {sim_count} ({sim_count/len(df):.1%})")
        print(f"   - Classe 'Não': {nao_count} ({nao_count/len(df):.1%})")
        
    except Exception as e:
        print(f"❌ Erro no processamento: {str(e)}")

if __name__ == "__main__":
    main()