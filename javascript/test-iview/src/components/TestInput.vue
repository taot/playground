<template>
    <h2>Test Input</h2>
    <i-input :value.sync="value" placeholder="请输入..." style="width: 300px"></i-input>
    <i-input>
        <i-select :model.sync="select1" slot="prepend" style="width: 80px">
            <i-option value="http">http://</i-option>
            <i-option value="https">https://</i-option>
        </i-select>
        <i-select :model.sync="select2" slot="append" style="width: 70px">
            <i-option value="com">.com</i-option>
            <i-option value="org">.org</i-option>
            <i-option value="io">.io</i-option>
        </i-select>
    </i-input>
</template>

<script>
export default {
    data() {
        return {
            value: "",
            select1: "http",
            select2: "com"
        }
    }
}
</script>

<style>
</style>
