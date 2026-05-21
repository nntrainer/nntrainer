import * as vscode from 'vscode';

export class NodePropertiesProvider implements vscode.TreeDataProvider<PropertyItem> {
    private properties: Array<{ key: string; value: string }> = [];
    private _onDidChangeTreeData = new vscode.EventEmitter<PropertyItem | undefined>();
    readonly onDidChangeTreeData = this._onDidChangeTreeData.event;

    setProperties(props: Array<{ key: string; value: string }>) {
        this.properties = props;
        this._onDidChangeTreeData.fire(undefined);
    }

    getTreeItem(element: PropertyItem): vscode.TreeItem {
        return element;
    }

    getChildren(): PropertyItem[] {
        if (this.properties.length === 0) {
            return [new PropertyItem('Select a node', '')];
        }
        return this.properties.map(p => new PropertyItem(p.key, p.value));
    }
}

class PropertyItem extends vscode.TreeItem {
    constructor(key: string, value: string) {
        super(value ? `${key}: ${value}` : key, vscode.TreeItemCollapsibleState.None);
        this.description = value ? undefined : 'no selection';
    }
}
