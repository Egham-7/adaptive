interface AttributeProps {
    name: string;
    size: number;
    data?: any;
}
interface UniformProps {
    type: string;
    value: Array<number>;
    location?: WebGLUniformLocation;
}
interface GeometryProps {
    vertices?: Array<Array<object>>;
    normal?: Array<Array<object>>;
}
interface BufferProps {
    location: number;
    buffer: WebGLBuffer;
    size: number;
}
interface InstanceProps {
    attributes?: Array<AttributeProps>;
    vertex?: string;
    fragment?: string;
    geometry?: GeometryProps;
    mode?: number;
    modifiers?: object;
    multiplier?: number;
    uniforms: {
        [key: string]: UniformProps;
    };
}
interface RendererProps {
    canvas?: HTMLCanvasElement;
    context?: object;
    contextType?: string;
    settings?: object;
}
/**
 * Class representing an instance.
 */
declare class Instance {
    gl: WebGLRenderingContext;
    vertex: string;
    fragment: string;
    program: WebGLProgram;
    uniforms: {
        [key: string]: UniformProps;
    };
    geometry: GeometryProps;
    attributes: Array<AttributeProps>;
    attributeKeys: Array<string>;
    multiplier: number;
    modifiers: Array<Function>;
    buffers: Array<BufferProps>;
    uniformMap: object;
    mode: number;
    onRender?: Function;
    /**
     * Create an instance.
     */
    constructor(props: InstanceProps);
    /**
     * Compile a shader.
     */
    compileShader(type: number, source: string): WebGLShader;
    /**
     * Create a program.
     */
    prepareProgram(): void;
    /**
     * Create uniforms.
     */
    prepareUniforms(): void;
    /**
     * Create buffer attributes.
     */
    prepareAttributes(): void;
    /**
     * Prepare a single attribute.
     */
    prepareAttribute(attribute: AttributeProps): void;
    /**
     * Create a buffer with an attribute.
     */
    prepareBuffer(attribute: AttributeProps): void;
    /**
     * Render the instance.
     */
    render(renderUniforms: object): void;
    /**
     * Destroy the instance.
     */
    destroy(): void;
}
/**
 * Class representing a Renderer.
 */
declare class Renderer {
    clearColor: Array<GLclampf>;
    onRender: Function;
    onSetup: Function;
    uniformMap: object;
    gl: WebGLRenderingContext;
    canvas: HTMLCanvasElement;
    devicePixelRatio: number;
    clip: Array<number>;
    instances: Map<string, Instance>;
    position: {
        x: number;
        y: number;
        z: number;
    };
    uniforms: {
        [key: string]: UniformProps;
    };
    shouldRender: boolean;
    /**
     * Create a renderer.
     */
    constructor(props: RendererProps);
    /**
     * Handle resize events.
     */
    resize(): void;
    /**
     * Toggle the active state of the renderer.
     */
    toggle(shouldRender: boolean): void;
    /**
     * Render the total scene.
     */
    render(): void;
    /**
     * Add an instance to the renderer.
     */
    add(key: string, settings: InstanceProps): Instance;
    /**
     * Remove an instance from the renderer.
     */
    remove(key: string): void;
    /**
     * Destroy the renderer and its instances.
     */
    destroy(): void;
}
export default Renderer;
