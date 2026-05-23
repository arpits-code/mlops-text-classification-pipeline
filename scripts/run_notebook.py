import sys
import nbformat

try:
    from nbclient import NotebookClient
except Exception as e:
    print("NBCLIENT_MISSING:", e)
    sys.exit(2)


def main():
    if len(sys.argv) < 3:
        print("USAGE: python scripts/run_notebook.py <input_ipynb> <output_ipynb>")
        sys.exit(1)
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    try:
        nb = nbformat.read(input_path, as_version=4)
    except Exception as e:
        print("READ_FAILED:", e)
        sys.exit(3)
    try:
        client = NotebookClient(nb, timeout=600)
        client.execute()
        nbformat.write(nb, output_path)
        print("NOTEBOOK_EXECUTED")
        return
    except Exception as e:
        print("NBCLIENT_EXECUTION_FAILED, falling back to cell-by-cell exec:", e)

    # Fallback: execute code cells sequentially in the current Python process
    exec_globals = {}
    for idx, cell in enumerate(nb.cells):
        try:
            if cell.get('cell_type') != 'code':
                continue
            source = cell.get('source', '')
            if isinstance(source, list):
                source = "\n".join(source)
            compiled = compile(source, f"<cell {idx}>", 'exec')
            exec(compiled, exec_globals)
            print(f"CELL_EXECUTED:{idx}")
        except Exception as e:
            print(f"CELL_FAILED:{idx}", e)
            sys.exit(5)

    # Write the notebook back (no outputs added in fallback)
    try:
        nbformat.write(nb, output_path)
    except Exception:
        pass
    print("NOTEBOOK_EXECUTED_FALLBACK")

if __name__ == '__main__':
    main()
