<template>
    <div id="app">
        <Alert type="success" show-icon closable>成功提示文案</Alert>
        <Row>
            <i-col span="12">
                <test-button></test-button>
            </i-col>
            <i-col span="12">
                <test-input></test-input>
            </i-col>
        </Row>
        <Row>
            <i-col span="12">
                <!-- <test-button></test-button> -->
            </i-col>
            <i-col span="12">
                <!-- <test-button></test-button> -->
            </i-col>
        </Row>

    </div>
</template>

<script>
import TestButton from './components/TestButton'
import TestInput from './components/TestInput'

export default {
    components: {
        TestButton,
        TestInput
    }
}
</script>

<style>
html {
    height: 100%;
}

body {
    display: flex;
    /*align-items: center;*/
    justify-content: center;
    height: 100%;
}

#app {
    color: #2c3e50;
    /*margin-top: -100px;*/
    /*max-width: 600px;*/
    width: 100%;
    font-family: Source Sans Pro, Helvetica, sans-serif;
    text-align: center;
}

#app a {
    color: #42b983;
    text-decoration: none;
}

.logo {
    width: 100px;
    height: 100px
}
</style>
