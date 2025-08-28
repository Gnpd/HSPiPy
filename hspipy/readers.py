import pandas as pd
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Union
from abc import ABC, abstractmethod


class HSPReader(ABC):
    """Abstract base class for HSP data readers."""
    
    @abstractmethod
    def read(self, path: Union[str, Path]) -> pd.DataFrame:
        """Read HSP data from file and return standardized DataFrame."""
        pass
    
    def _validate_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and standardize the DataFrame structure."""
        required_columns = ['Solvent', 'D', 'P', 'H', 'Score']
        
        # Check for required columns
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Select only required columns
        df = df[required_columns].copy()
        
        # Convert numeric columns
        numeric_columns = ['D', 'P', 'H', 'Score']
        for col in numeric_columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Handle Score column specially (can contain "-" or other non-numeric values)
        if not pd.api.types.is_numeric_dtype(df['Score']):
            # Replace common non-numeric placeholders with NaN, then convert to numeric
            df['Score'] = df['Score'].replace(['-', '', 'N/A', 'n/a', 'NA'], np.nan)
            df['Score'] = pd.to_numeric(df['Score'], errors='coerce')

        # Remove rows where D, P, H are NaN
        initial_count = len(df)
        df = df.dropna(subset=['D', 'P', 'H'])
        if len(df) < initial_count:
            print(f"Warning: Removed {initial_count - len(df)} rows with invalid data")
        
        if len(df) == 0:
            raise ValueError("No valid data rows found after cleaning")
        
        # Report missing Score values but don't modify them
        score_missing = df['Score'].isna().sum()
        if score_missing > 0:
            print(f"Info: {score_missing} rows have missing Score values (will be excluded during fitting)")
             
        # Validate HSP parameter ranges
        self._validate_hsp_ranges(df)
        
        return df
    
    def _validate_hsp_ranges(self, df: pd.DataFrame):
        """Validate HSP parameter ranges and issue warnings."""
        for param, col in [('D', 'D'), ('P', 'P'), ('H', 'H')]:
            out_of_range = ((df[col] < 0) | (df[col] > 50)).sum()
            if out_of_range > 0:
                print(f"Warning: {out_of_range} {param} values outside typical range (0-50)")


class CSVReader(HSPReader):
    """Reader for CSV files with or without headers."""
    
    def read(self, path: Union[str, Path]) -> pd.DataFrame:
        """Read CSV file, handling both with and without headers."""
        try:
            # First, try reading with headers
            df = pd.read_csv(path)
            
            # Check if we have the expected columns
            expected_cols = ['Solvent', 'D', 'P', 'H', 'Score']
            if all(col in df.columns for col in expected_cols):
                return self._validate_dataframe(df)
            
            # If not, assume no headers and use default column names
            df = pd.read_csv(path, header=None)
            if df.shape[1] >= 5:
                df.columns = ['Solvent', 'D', 'P', 'H', 'Score'] + [f'Extra_{i}' for i in range(5, df.shape[1])]
                return self._validate_dataframe(df)
            else:
                raise ValueError(f"CSV file must have at least 5 columns, found {df.shape[1]}")
                
        except pd.errors.EmptyDataError:
            raise ValueError("CSV file is empty")
        except Exception as e:
            raise ValueError(f"Error reading CSV file: {str(e)}")


class HSDReader(HSPReader):
    """Reader for HSD (tab-separated) files."""
    
    def read(self, path: Union[str, Path]) -> pd.DataFrame:
        """Read HSD (tab-separated) file."""
        try:
            # Read tab-separated file
            df = pd.read_csv(path, sep='\t')
            
            # Map common column variations to standard names
            column_mapping = {
                'dD': 'D', 'δD': 'D', 'Delta_D': 'D',
                'dP': 'P', 'δP': 'P', 'Delta_P': 'P', 
                'dH': 'H', 'δH': 'H', 'Delta_H': 'H'
            }
            
            df = df.rename(columns=column_mapping)
            return self._validate_dataframe(df)
            
        except Exception as e:
            raise ValueError(f"Error reading HSD file: {str(e)}")


class HSDXReader(HSPReader):
    """Reader for HSDX (XML) files."""
    
    def read(self, path: Union[str, Path]) -> pd.DataFrame:
        """Read HSDX (XML) file."""
        try:
            tree = ET.parse(path)
            root = tree.getroot()
            
            data = []
            for chemical in root.findall('.//Chemical'):
                # Extract data from XML elements
                solvent = self._get_xml_text(chemical, 'Solvent')
                d_val = self._get_xml_text(chemical, 'δD')
                p_val = self._get_xml_text(chemical, 'δP') 
                h_val = self._get_xml_text(chemical, 'δH')
                score = self._get_xml_text(chemical, 'Score')
                
                if all(val is not None for val in [solvent, d_val, p_val, h_val, score]):
                    try:
                        data.append({
                            'Solvent': solvent,
                            'D': float(d_val),
                            'P': float(p_val),
                            'H': float(h_val),
                            'Score': float(score)
                        })
                    except ValueError:
                        # Skip rows with invalid numeric data
                        continue
            
            if not data:
                raise ValueError("No valid chemical data found in HSDX file")
                
            df = pd.DataFrame(data)
            return self._validate_dataframe(df)
            
        except ET.ParseError as e:
            raise ValueError(f"Invalid XML format in HSDX file: {str(e)}")
        except Exception as e:
            raise ValueError(f"Error reading HSDX file: {str(e)}")
    
    def _get_xml_text(self, element, tag):
        """Safely extract text from XML element."""
        child = element.find(tag)
        return child.text if child is not None else None


class HSPDataReader:
    """Main reader class that handles multiple formats."""
    
    def __init__(self):
        self._readers = {
            '.csv': CSVReader(),
            '.hsd': HSDReader(),
            '.hsdx': HSDXReader()
        }
    
    def read(self, path: Union[str, Path]) -> pd.DataFrame:
        """
        Read HSP data from various file formats.
        
        Parameters
        ----------
        path : str or Path
            Path to the HSP data file
            
        Returns
        -------
        pd.DataFrame
            Standardized DataFrame with columns: Solvent, D, P, H, Score
            
        Raises
        ------
        FileNotFoundError
            If the file doesn't exist
        ValueError
            If the file format is not supported or data is invalid
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"HSP data file not found: {path}")
        
        file_ext = path.suffix.lower()
        
        # Get appropriate reader
        if file_ext in self._readers:
            reader = self._readers[file_ext]
        else:
            # Try auto-detection
            reader = self._auto_detect_reader(path)
        
        try:
            df = reader.read(path)
            print(f"Successfully loaded {len(df)} solvent records from {path.name}")
            return df
        except Exception as e:
            raise ValueError(f"Failed to read HSP data from {path}: {str(e)}")
    
    def _auto_detect_reader(self, path: Path) -> HSPReader:
        """Auto-detect file format based on content."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
            
            # Check if it's XML
            if first_line.startswith('<?xml') or first_line.startswith('<'):
                return self._readers['.hsdx']
            
            # Check if it's tab-separated (likely HSD)
            if '\t' in first_line:
                return self._readers['.hsd']
            
            # Default to CSV
            return self._readers['.csv']
            
        except Exception:
            # If auto-detection fails, default to CSV
            return self._readers['.csv']
    
    def register_reader(self, extension: str, reader: HSPReader):
        """Register a custom reader for a specific file extension."""
        self._readers[extension.lower()] = reader